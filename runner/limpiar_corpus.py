#!/usr/bin/env python3
"""
limpiar_corpus.py (v3) — Limpieza dirigida del corpus OCR (corpus.csv).

v1 (auditoría sobre peores_ocr.csv):
  * <br> y tags HTML residuales de docling; entidades.
  * Tablas markdown: elimina separadores (|---|---|), convierte contenido a
    texto plano (pipes -> espacio) y marca la fila con flag `tabla`.
  * Líneas separadoras basura (******, ------) y rachas de símbolos.
  * '##' residuales; des-guionado ("pala- bra" -> "palabra").
  * Corrección conservadora 0/1 dentro de palabras ("Uni0n" -> "Union");
    nunca toca teléfonos, códigos ni cifras.

v2 (inspección de los raw.md de docling):
  * Letras capitales perdidas (la capitular decorativa sale como imagen):
    "OMINGO"->"DOMINGO", "ENVENIDA"->"BIENVENIDA", "eportes"->"Deportes",
    "Il mundo"->"El mundo", "L UNIVERSO"->"EL UNIVERSO". Solo palabras
    inequívocas (nada que sea también palabra válida en español).
  * Mobiliario de portada/cabecera: líneas de precio (PVP ... IVA), edición y
    paginación (PRIMERA EDICIÓN ... SECCIONES ... PÁGS., AÑO: 91, Nº 108),
    lemas de cabecera y URLs sueltas.
  * Códigos de barras leídos como texto (líneas de solo 0/1).
  * Títulos basura: si el título es solo dígitos/símbolos (p.ej. el código de
    barras como '## 0110100...'), se vacía y se cuenta.

v3 (revisión pre-cobertura, para no perder señal):
  * `vacio` ahora exige titulo Y texto vacíos. Las filas con titular pero sin
    cuerpo (portadas, teasers) llevan el flag `solo_titulo` y SOBREVIVEN a
    --drop vacio: los titulares son señal densa para el índice de nowcasting.
    (En v2, --drop vacio eliminaba estas filas: 54k en el corpus completo.)
  * Mobiliario de cabecera más estricto: los patrones de EDICIÓN/SECCIONES/PÁGS
    ahora exigen dígitos ("6 SECCIONES", "156 PÁGS."), para no tocar prosa que
    mencione esas palabras.

Flags por fila (columna `flags`, sobre el texto ORIGINAL, separados por ';'):
  tabla | numerico | clasificado | solo_titulo | vacio

Por defecto NO se elimina ninguna fila; --drop excluye categorías:
    --drop vacio
    --drop vacio,tabla,numerico,clasificado

Uso:
  python3 limpiar_corpus.py --in csv_corpus/corpus.csv \
      --out csv_corpus/corpus_limpio.csv [--drop vacio] [--max-filas 0]
"""
import argparse
import csv
import html
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

csv.field_size_limit(sys.maxsize)

# ── Regex de limpieza (v1) ──────────────────────────────────────────────────
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]{0,60}>")
ENT_RE = re.compile(r"&[a-zA-Z#][a-zA-Z0-9]{1,8};")
SEP_LINE_RE = re.compile(r"^\s*[\*\-=_•·.…#~|<>]{4,}\s*$")
RUN_RE = re.compile(r"([*=_•·#~<>])\1{3,}")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")
HASH_RE = re.compile(r"#{2,}")
GUION_RE = re.compile(r"([a-záéíóúüñ])-\s+([a-záéíóúüñ])")
ESPACIOS_RE = re.compile(r"[ \t]{2,}")

LETRAS = "A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
MIXWORD_RE = re.compile(
    rf"(?<![\w])(?=[{LETRAS}01]*[01])(?=[{LETRAS}01]*[{LETRAS}])[{LETRAS}01]+(?![\w])"
)
VOCALES = set("aáeéiíoóuúü")
CONS5_RE = re.compile(r"[bcdfghjklmnñpqrstvwxyz]{5,}", re.IGNORECASE)

# ── v2: mobiliario de cabecera, códigos de barras, capitulares ─────────────
FURNITURE_RES = [
    re.compile(r"PVP\s+FINAL|INCLUIDO\s+IVA", re.IGNORECASE),
    re.compile(r"\bEDICI[ÓO]N\b.{0,40}\b\d+\s+SECCIONES\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+SECCIONES\b.{0,25}\b\d+\s*P[ÁA]GS?\b", re.IGNORECASE),
    re.compile(r"\bAÑO:\s*\d+\s*,?\s*Nº", re.IGNORECASE),
    re.compile(r"^\s*/?www\.\S+\s*$", re.IGNORECASE),
    re.compile(r"^\s*EL\s+MAYOR\s+DIARIO\s+NACIONAL\s*$", re.IGNORECASE),
    re.compile(r"^\s*DIRECCI[ÓO]N\s+EN\s+INTERNET\b", re.IGNORECASE),
]
BARCODE_LINE_RE = re.compile(r"^\s*[01]{12,}\s*$")

DROPCAP = {
    "OMINGO": "DOMINGO", "IÉRCOLES": "MIÉRCOLES", "IERCOLES": "MIERCOLES",
    "UEVES": "JUEVES", "IERNES": "VIERNES", "ÁBADO": "SÁBADO",
    "ENVENIDA": "BIENVENIDA", "IENVENIDA": "BIENVENIDA",
    "EPORTES": "DEPORTES", "eportes": "Deportes",
    "CONOMÍA": "ECONOMÍA", "conomía": "Economía",
    "NTERNACIONAL": "INTERNACIONAL", "nternacional": "Internacional",
    "CTUALIDAD": "ACTUALIDAD", "ctualidad": "Actualidad",
    "SPECTÁCULOS": "ESPECTÁCULOS", "spectáculos": "Espectáculos",
    "NTRETENIMIENTO": "ENTRETENIMIENTO", "ntretenimiento": "Entretenimiento",
}
DROPCAP_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(k) for k in DROPCAP) + r")(?![\w])"
)
MASTHEAD_L_RE = re.compile(r"\bL\s+(COMERCIO|UNIVERSO|TEL[ÉE]GRAFO)\b")
IL_EL_RE = re.compile(r"\bIl(?=\s+[a-záéíóúüñ])")

TITULO_BASURA_RE = re.compile(r"^[\d\s.:,;•·*\-_/|]{6,}$")

# ── Flags ───────────────────────────────────────────────────────────────────
COD_CLASIF_RE = re.compile(r"[\(\[]\s*\d{8,}\s*[\)\]]")
TEL_RE = re.compile(r"\b(?:Telf?s?\.?|Tel[eé]fonos?|Informes|Llamar)\b", re.IGNORECASE)


def es_palabra_plausible(tok: str) -> bool:
    low = tok.lower()
    if not low.isalpha() or not (2 <= len(low) <= 24):
        return False
    if not any(c in VOCALES for c in low):
        return False
    if CONS5_RE.search(low):
        return False
    return True


def corregir_mixto(m: re.Match, stats: Dict[str, int]) -> str:
    tok = m.group(0)
    letras = sum(c.isalpha() for c in tok)
    digs = len(tok) - letras
    if letras < 3 or digs > 2:
        return tok
    for repl in ({"0": "o", "1": "l"}, {"0": "o", "1": "i"}):
        cand = "".join(repl.get(c, c) for c in tok)
        if tok.isupper():
            cand = cand.upper()
        if es_palabra_plausible(cand):
            stats["tokens_corregidos"] += 1
            return cand
    return tok


def corregir_capitulares(t: str, stats: Dict[str, int]) -> str:
    def _sub(m: re.Match) -> str:
        stats["capitulares"] += 1
        return DROPCAP[m.group(1)]
    t = DROPCAP_RE.sub(_sub, t)
    t, n1 = MASTHEAD_L_RE.subn(r"EL \1", t)
    t, n2 = IL_EL_RE.subn("El", t)
    stats["capitulares"] += n1 + n2
    return t


def limpiar_titulo(titulo: str, stats: Dict[str, int]) -> str:
    t = ESPACIOS_RE.sub(
        " ", HASH_RE.sub(" ", TAG_RE.sub(" ", html.unescape(titulo)))).strip()
    if t and (TITULO_BASURA_RE.match(t) or not re.search(rf"[{LETRAS}]", t)):
        stats["titulos_basura"] += 1
        return ""
    return corregir_capitulares(t, stats)


def limpiar_texto(texto: str, stats: Dict[str, int]) -> Tuple[str, bool]:
    """Devuelve (texto_limpio, tenia_tabla)."""
    t = html.unescape(texto)
    t = BR_RE.sub(" ", t)
    t = TAG_RE.sub(" ", t)
    t = ENT_RE.sub(" ", t)

    tenia_tabla = False
    lineas: List[str] = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            lineas.append("")
            continue
        if TABLE_SEP_RE.match(s):
            tenia_tabla = True
            stats["lineas_sep_tabla"] += 1
            continue
        if SEP_LINE_RE.match(s):
            stats["lineas_basura"] += 1
            continue
        if BARCODE_LINE_RE.match(s):
            stats["lineas_basura"] += 1
            continue
        if any(rx.search(s) for rx in FURNITURE_RES):
            stats["lineas_cabecera"] += 1
            continue
        if s.count("|") >= 2:
            tenia_tabla = True
            stats["lineas_tabla_convertidas"] += 1
        ln = ln.replace("|", " ")
        ln = RUN_RE.sub(" ", ln)
        ln = HASH_RE.sub(" ", ln)
        lineas.append(ln)
    t = "\n".join(lineas)

    t = GUION_RE.sub(r"\1\2", t)
    t = corregir_capitulares(t, stats)
    t = MIXWORD_RE.sub(lambda m: corregir_mixto(m, stats), t)
    t = ESPACIOS_RE.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip(), tenia_tabla


def flags_fila(texto_orig: str, texto_limpio: str, titulo_limpio: str,
               tenia_tabla: bool) -> List[str]:
    fl: List[str] = []
    n = len(texto_orig)
    if tenia_tabla or texto_orig.count("|") >= 4:
        fl.append("tabla")
    if n >= 30:
        digs = sum(c.isdigit() for c in texto_orig)
        if digs / n > 0.25:
            fl.append("numerico")
    if (len(COD_CLASIF_RE.findall(texto_orig)) >= 2
            or (len(TEL_RE.findall(texto_orig)) >= 2
                and n and sum(c.isdigit() for c in texto_orig) / n > 0.08)):
        fl.append("clasificado")
    if not texto_limpio:
        if titulo_limpio:
            fl.append("solo_titulo")   # titular sin cuerpo: se conserva
        else:
            fl.append("vacio")         # nada aprovechable
    return fl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="",
                    help="flags a excluir, separados por coma (p.ej. vacio,tabla)")
    ap.add_argument("--max-filas", type=int, default=0)
    args = ap.parse_args()
    drop: Set[str] = {s.strip() for s in args.drop.split(",") if s.strip()}

    stats: Dict[str, int] = {
        "tokens_corregidos": 0, "lineas_sep_tabla": 0,
        "lineas_tabla_convertidas": 0, "lineas_basura": 0,
        "lineas_cabecera": 0, "capitulares": 0, "titulos_basura": 0,
    }
    conteo_flags: Dict[str, int] = {}
    filas = escritas = descartadas = modificadas = 0

    with Path(args.inp).open("r", encoding="utf-8-sig", newline="") as fi, \
         Path(args.out).open("w", newline="", encoding="utf-8") as fo:
        rd = csv.DictReader(fi)
        campos = list(rd.fieldnames or [])
        if "flags" not in campos:
            campos.append("flags")
        w = csv.DictWriter(fo, fieldnames=campos)
        w.writeheader()
        for row in rd:
            filas += 1
            orig = row.get("texto", "") or ""
            limpio, tenia_tabla = limpiar_texto(orig, stats)
            row["titulo"] = limpiar_titulo(row.get("titulo", "") or "", stats)
            fl = flags_fila(orig, limpio, row["titulo"], tenia_tabla)
            for f in fl:
                conteo_flags[f] = conteo_flags.get(f, 0) + 1
            if limpio != orig:
                modificadas += 1
            row["texto"] = limpio
            row["flags"] = ";".join(fl)
            if drop and any(f in drop for f in fl):
                descartadas += 1
            else:
                w.writerow(row)
                escritas += 1
            if args.max_filas and filas >= args.max_filas:
                break
            if filas % 200000 == 0:
                print(f"  {filas} filas...", flush=True)

    det = " | ".join(f"{k} {v}" for k, v in sorted(conteo_flags.items()))
    print(f"[ok] {filas} filas -> {escritas} escritas "
          f"({descartadas} descartadas por --drop {sorted(drop) or '∅'})")
    print(f"     modificadas: {modificadas} | flags: {det or 'ninguno'}")
    print(f"     tokens 0/1: {stats['tokens_corregidos']} | "
          f"capitulares: {stats['capitulares']} | "
          f"títulos basura: {stats['titulos_basura']}")
    print(f"     líneas — tabla convertidas: {stats['lineas_tabla_convertidas']} | "
          f"separadores: {stats['lineas_sep_tabla']} | "
          f"basura: {stats['lineas_basura']} | "
          f"cabecera/portada: {stats['lineas_cabecera']}")
    print(f"OUT: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())