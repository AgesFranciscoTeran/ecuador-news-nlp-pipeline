#!/usr/bin/env python3
"""
empalmar_csv.py (v2) — Une el corpus de webscraping y el corpus OCR (limpio)
en UN corpus maestro, con DEDUPLICACIÓN real en los días de solape.

Por qué no dedup por día entero: en los días donde ambas fuentes cubren a
El Universo, la web trae ~61 artículos/día y el impreso ~344 secciones/día —
descartar el impreso completo botaría todo su contenido único (páginas que la
web nunca publicó). La política correcta es por ARTÍCULO:

  * Solo se evalúan filas OCR "tipo artículo" (sin flags tabla/numerico/
    clasificado) cuyo (periodico, fecha) también existe en la web.
  * El titular OCR se compara contra los titulares web de ese mismo día:
    tokens en minúscula, sin tildes ni stopwords; similitud = containment
    |A∩B| / min(|A|,|B|). Si ≥ --umbral (0.6) con algún titular web, la fila
    OCR se descarta como duplicado (la versión web es más limpia).
  * Los flujos con flag (clasificado, numerico, tabla) NUNCA se deduplican:
    la web no tiene contraparte y son señal para el nowcasting.
  * Filas sin titular utilizable (<3 tokens) se conservan (default seguro).

Esquema del maestro:
  id, origen, id_original, periodico, fecha, fecha_fuente, seccion,
  titulo, texto, flags, path
  - origen: webscraping | ocr ; id: consecutivo nuevo
  - fecha_fuente: "web" para webscraping; etiquetas del pipeline para OCR
  - seccion: solo webscraping ; flags/path: solo OCR

Uso:
  python3 empalmar_csv.py \
      --viejo csv_webscraping/articulos.csv \
      --nuevo csv_corpus/corpus_limpio.csv \
      --out csv_corpus/corpus_maestro.csv [--umbral 0.6] [--sin-dedup]

Streaming (dos pasadas sobre el CSV web: índice de titulares y volcado).
"""
import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

csv.field_size_limit(sys.maxsize)

YEAR_RANGE_RE = re.compile(r"\s*\d{4}\s*-\s*\d{4}\s*$")
DMY_RE = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$")
ISO_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {"de", "la", "el", "en", "los", "las", "del", "al", "un", "una", "y",
        "a", "o", "u", "e", "que", "con", "por", "para", "se", "su", "sus",
        "es", "no", "mas", "más", "lo", "le", "les"}
FLAGS_NO_DEDUP = {"clasificado", "numerico", "tabla"}

OUT_FIELDS = ["id", "origen", "id_original", "periodico", "fecha",
              "fecha_fuente", "seccion", "titulo", "texto", "flags", "path"]


def norm_periodico(name: str) -> str:
    return YEAR_RANGE_RE.sub("", (name or "").strip())


def norm_fecha(s: str) -> str:
    s = (s or "").strip()
    if ISO_RE.match(s):
        return s
    m = DMY_RE.match(s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return s


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def titulo_tokens(titulo: str) -> FrozenSet[str]:
    toks = TOKEN_RE.findall(sin_tildes((titulo or "").lower()))
    return frozenset(t for t in toks if t not in STOP and len(t) > 1)


def containment(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def lower_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {(k or "").strip().lower(): (v or "") for k, v in row.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viejo", required=True, help="CSV de webscraping")
    ap.add_argument("--nuevo", required=True, help="corpus_limpio.csv del OCR")
    ap.add_argument("--out", default="corpus_maestro.csv")
    ap.add_argument("--umbral", type=float, default=0.6,
                    help="containment mínimo para declarar duplicado")
    ap.add_argument("--sin-dedup", action="store_true")
    ap.add_argument("--max-filas", type=int, default=0)
    args = ap.parse_args()

    # ── Pasada A: índice de titulares web por (periodico, fecha) ───────────
    indice: Dict[Tuple[str, str], List[FrozenSet[str]]] = defaultdict(list)
    n_web = 0
    if not args.sin_dedup:
        with Path(args.viejo).open("r", encoding="utf-8-sig", newline="") as fh:
            for i, raw in enumerate(csv.DictReader(fh), 1):
                r = lower_keys(raw)
                per = norm_periodico(r.get("periodico", ""))
                fec = norm_fecha(r.get("fecha", ""))
                toks = titulo_tokens(r.get("titulo", ""))
                if per and ISO_RE.match(fec) and len(toks) >= 3:
                    indice[(per, fec)].append(toks)
                if args.max_filas and i >= args.max_filas:
                    break
        print(f"[..] índice web: {sum(len(v) for v in indice.values())} titulares "
              f"en {len(indice)} días-periódico", flush=True)

    next_id = 1
    dup = 0
    dup_por_per: Dict[str, int] = defaultdict(int)
    dias_con_dedup: Set[Tuple[str, str]] = set()
    escritas = {"webscraping": 0, "ocr": 0}
    fechas_raras = 0

    with Path(args.out).open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=OUT_FIELDS)
        w.writeheader()

        # ── Pasada B: volcar webscraping ───────────────────────────────────
        with Path(args.viejo).open("r", encoding="utf-8-sig", newline="") as fh:
            for i, raw in enumerate(csv.DictReader(fh), 1):
                r = lower_keys(raw)
                fec = norm_fecha(r.get("fecha", ""))
                if fec and not ISO_RE.match(fec):
                    fechas_raras += 1
                w.writerow({
                    "id": next_id, "origen": "webscraping",
                    "id_original": r.get("id", ""),
                    "periodico": norm_periodico(r.get("periodico", "")),
                    "fecha": fec, "fecha_fuente": "web",
                    "seccion": r.get("seccion", ""),
                    "titulo": r.get("titulo", ""),
                    "texto": r.get("texto", ""),
                    "flags": "", "path": "",
                })
                next_id += 1
                escritas["webscraping"] += 1
                if args.max_filas and i >= args.max_filas:
                    break

        # ── Pasada C: OCR con dedup por titular en días de solape ─────────
        with Path(args.nuevo).open("r", encoding="utf-8-sig", newline="") as fh:
            for i, raw in enumerate(csv.DictReader(fh), 1):
                r = lower_keys(raw)
                per = norm_periodico(r.get("periodico", ""))
                fec = norm_fecha(r.get("fecha", ""))
                flags = r.get("flags", "")
                flagset = {f for f in flags.split(";") if f}
                es_articulo = not (flagset & FLAGS_NO_DEDUP)
                if (not args.sin_dedup and es_articulo
                        and (per, fec) in indice):
                    toks = titulo_tokens(r.get("titulo", ""))
                    if len(toks) >= 3 and any(
                            containment(toks, wt) >= args.umbral
                            for wt in indice[(per, fec)]):
                        dup += 1
                        dup_por_per[per] += 1
                        dias_con_dedup.add((per, fec))
                        continue
                w.writerow({
                    "id": next_id, "origen": "ocr",
                    "id_original": r.get("id", ""),
                    "periodico": per, "fecha": fec,
                    "fecha_fuente": r.get("fecha_fuente", ""),
                    "seccion": "",
                    "titulo": r.get("titulo", ""),
                    "texto": r.get("texto", ""),
                    "flags": flags, "path": r.get("path", ""),
                })
                next_id += 1
                escritas["ocr"] += 1
                if i % 500000 == 0:
                    print(f"  ocr: {i} filas...", flush=True)
                if args.max_filas and i >= args.max_filas:
                    break

    total = escritas["webscraping"] + escritas["ocr"]
    print(f"[ok] maestro: {total} filas "
          f"(web {escritas['webscraping']} + ocr {escritas['ocr']}) -> {args.out}")
    if not args.sin_dedup:
        det = " | ".join(f"{p}: {n}" for p, n in sorted(dup_por_per.items()))
        print(f"     duplicados OCR eliminados: {dup} ({det or 'ninguno'}) "
              f"en {len(dias_con_dedup)} días-periódico | umbral {args.umbral}")
    if fechas_raras:
        print(f"[warn] {fechas_raras} fechas web no normalizables a ISO")
    return 0


if __name__ == "__main__":
    sys.exit(main())