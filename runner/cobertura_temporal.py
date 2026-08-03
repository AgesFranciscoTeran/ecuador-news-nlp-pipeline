#!/usr/bin/env python3
"""
cobertura_temporal.py — Mapa de cobertura temporal de ambos corpus, pensado
para nowcasting: antes de construir señales diarias hay que saber qué días
existen, con qué volumen, con qué confianza de fechado y dónde están los
huecos y el "costurón" entre fuentes.

Produce:
  * cobertura_diaria.csv — una fila por (fecha, fuente, periodico) con
    n_filas y n_chars. Lista para pivotar/graficar (pandas, Excel).
  * resumen impreso y resumen.txt:
      - por fuente y periódico: rango de fechas, días con datos, % del
        calendario cubierto, filas totales;
      - los 5 huecos más largos de cada serie (rachas de días sin datos);
      - distribución de fecha_fuente en el corpus OCR (header, propagación,
        retro, bloque_*): define qué granularidad es creíble — 'header' y
        'pagina_anterior' sostienen señal diaria; 'bloque_quincena/mes' solo
        mensual;
      - solape temporal entre webscraping y OCR por periódico (la zona donde
        hará falta deduplicar y donde el volumen cambia de régimen).

Uso:
  python3 cobertura_temporal.py \
      --viejo csv_webscraping/articulos.csv \
      --nuevo csv_corpus/corpus_limpio.csv \
      --out-dir cobertura [--max-filas 0]
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(sys.maxsize)

YEAR_RANGE_RE = re.compile(r"\s*\d{4}\s*-\s*\d{4}\s*$")
DMY_RE = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$")
ISO_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")


def norm_periodico(name: str) -> str:
    return YEAR_RANGE_RE.sub("", (name or "").strip())


def parse_fecha(s: str) -> Optional[date]:
    s = (s or "").strip()
    m = ISO_RE.match(s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = DMY_RE.match(s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def lower_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {(k or "").strip().lower(): (v or "") for k, v in row.items()}


def procesar(path: Path, fuente: str, acum: Dict, fechas_malas: Counter,
             ff_counter: Counter, max_filas: int) -> int:
    n = 0
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            n += 1
            r = lower_keys(raw)
            dt = parse_fecha(r.get("fecha", ""))
            per = norm_periodico(r.get("periodico", "")) or "?"
            if fuente == "ocr":
                ff_counter[r.get("fecha_fuente", "") or "?"] += 1
            if dt is None:
                fechas_malas[fuente] += 1
            else:
                cel = acum[(dt, fuente, per)]
                cel[0] += 1
                cel[1] += len(r.get("texto", "") or "")
            if max_filas and n >= max_filas:
                break
            if n % 500000 == 0:
                print(f"  {fuente}: {n} filas...", flush=True)
    return n


def huecos(dias: List[date], top: int = 5) -> List[Tuple[int, date, date]]:
    """Top rachas de días SIN datos dentro del rango de la serie."""
    out: List[Tuple[int, date, date]] = []
    for a, b in zip(dias, dias[1:]):
        g = (b - a).days - 1
        if g > 0:
            out.append((g, a + timedelta(days=1), b - timedelta(days=1)))
    return sorted(out, reverse=True)[:top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viejo", required=True, help="CSV de webscraping")
    ap.add_argument("--nuevo", required=True, help="corpus (limpio) del OCR")
    ap.add_argument("--out-dir", default="cobertura")
    ap.add_argument("--max-filas", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    acum: Dict[Tuple[date, str, str], List[int]] = defaultdict(lambda: [0, 0])
    fechas_malas: Counter = Counter()
    ff: Counter = Counter()

    for fuente, path in (("webscraping", Path(args.viejo)),
                         ("ocr", Path(args.nuevo))):
        print(f"[..] {fuente}: {path}", flush=True)
        n = procesar(path, fuente, acum, fechas_malas, ff, args.max_filas)
        print(f"     {n} filas leídas", flush=True)

    # ── CSV diario ──────────────────────────────────────────────────────────
    with (out / "cobertura_diaria.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fecha", "fuente", "periodico", "n_filas", "n_chars"])
        for (dt, fuente, per), (nf, nc) in sorted(acum.items()):
            w.writerow([dt.isoformat(), fuente, per, nf, nc])

    # ── Resumen ─────────────────────────────────────────────────────────────
    L: List[str] = []
    series: Dict[Tuple[str, str], List[date]] = defaultdict(list)
    for (dt, fuente, per) in acum:
        series[(fuente, per)].append(dt)

    L.append("== Cobertura por fuente y periódico ==")
    for (fuente, per), dias in sorted(series.items()):
        dias.sort()
        d0, d1 = dias[0], dias[-1]
        rango = (d1 - d0).days + 1
        filas = sum(acum[(d, fuente, per)][0] for d in dias)
        L.append(f"  {fuente:<12} {per:<14} {d0} → {d1} | "
                 f"{len(dias)}/{rango} días ({100.0*len(dias)/rango:.1f}%) | "
                 f"{filas} filas | mediana filas/día "
                 f"{sorted(acum[(d, fuente, per)][0] for d in dias)[len(dias)//2]}")
        for g, a, b in huecos(dias):
            L.append(f"      hueco de {g} días: {a} → {b}")

    L.append("\n== fecha_fuente (OCR): qué granularidad sostiene ==")
    tot = sum(ff.values()) or 1
    for k, v in ff.most_common():
        L.append(f"  {k:<18} {v:>10} ({100.0*v/tot:.1f}%)")

    L.append("\n== Solape temporal webscraping ∩ OCR (dedup necesaria aquí) ==")
    pers = sorted({p for (_, p) in series})
    for per in pers:
        a = series.get(("webscraping", per))
        b = series.get(("ocr", per))
        if a and b:
            ini = max(min(a), min(b))
            fin = min(max(a), max(b))
            if ini <= fin:
                L.append(f"  {per:<14} {ini} → {fin} "
                         f"({(fin-ini).days+1} días de solape)")
            else:
                L.append(f"  {per:<14} sin solape")
        else:
            L.append(f"  {per:<14} presente solo en "
                     f"{'webscraping' if a else 'ocr'}")

    if fechas_malas:
        L.append("\n== Fechas no parseables (excluidas del mapa) ==")
        for k, v in fechas_malas.items():
            L.append(f"  {k}: {v}")

    resumen = "\n".join(L)
    print("\n" + resumen)
    (out / "resumen.txt").write_text(resumen + "\n", encoding="utf-8")
    print(f"\n[ok] {out / 'cobertura_diaria.csv'} y {out / 'resumen.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())