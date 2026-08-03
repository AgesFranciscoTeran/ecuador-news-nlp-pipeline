#!/usr/bin/env python3
"""
corpus_to_csv.py (v3.1) — UNA pasada por raw.md de los 3 periódicos, FECHADO 100%.

v2: el día de la semana se usa como checksum en el scoring del header
    (+2 presente, +8 consistente, −12 inconsistente).

v3 — RESCATE DE ANCLAS EN EL CUERPO (desapilar bloques colapsados):
  Diagnóstico: bloques mes/quincena enteros sin ninguna ancla de header
  colapsaban a su día inicial (p.ej. El Comercio oct-2019: 7.163 filas en
  2019-10-01), leyéndose como "huecos" que empiezan el día 2 o el 17.
  Muchas de esas páginas SÍ traen la fecha en el cuerpo (folio/pie).

  La v3, para toda página SIN ancla de header, busca la fecha en el texto
  completo exigiendo TRES condiciones simultáneas:
    1) día de la semana PRESENTE (coma opcional: "viernes, 11 de octubre..."),
    2) día de la semana CONSISTENTE con la fecha (checksum 1/7),
    3) fecha DENTRO del rango del bloque (quincena/mes/año de la carpeta).
  Y un cuarto filtro a nivel de bloque: MONOTONÍA — las páginas van en orden
  de escaneo, así que un ancla de cuerpo solo se acepta si no retrocede
  respecto de la última ancla aceptada ni supera la siguiente ancla de
  header. Esto descarta la trampa clásica: eventos futuros anunciados con
  fecha y día de semana correctos ("la marcha será el lunes 28 de octubre").

  Nuevas etiquetas de fecha_fuente (confianza intermedia, auditable):
    cuerpo        — ancla rescatada del cuerpo de la página
    cuerpo_ant    — heredada hacia adelante desde un ancla de cuerpo
    cuerpo_retro  — rellenada hacia atrás desde un ancla de cuerpo
  Las etiquetas existentes (header, pagina_anterior, retro, bloque_*) no
  cambian de significado. Capa diaria estricta: {header, pagina_anterior,
  retro}; extendida: + {cuerpo, cuerpo_ant, cuerpo_retro}; mensual: todas.

Pipeline sin cambios en lo demás:
  fechado header (scoring) > cuerpo > propagación > retro > inicio de bloque;
  limpieza soft_clean_raw; segmentación por '##';
  CSV: id, periodico, path, fecha, fecha_fuente, titulo, texto
"""
import argparse
import calendar
import csv
import html
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEF_IN_ROOT = "/home/fteran/Proyecto_DataHub"
DEF_OUT_DIR = "/home/fteran/Proyecto_DataHub/csv_corpus"
PAPERS = [
    "El Comercio 2001-2022",
    "El Telégrafo 2001-2019",
    "El Universo 2001-2022",
]
OUT_CSV = "corpus.csv"

# ── Fechas: léxico y regex ──────────────────────────────────────────────────
MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}
WEEKDAYS = {
    "lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2, "jueves": 3,
    "viernes": 4, "sabado": 5, "sábado": 5, "domingo": 6,
}
MONTH_ALT = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
WD_ALT = "lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo"
CORE_DATE = rf"""
    (?P<day>\d{{1,2}}|[Il1]\d|\d[Il1]|[Il1]{{2}})
    \s+de\s+
    (?P<month>{MONTH_ALT})
    \s+d(?:e|el)\s+
    (?P<year>\d{{3,4}}|[12][09OIl]\d{{2}})
"""

DATE_REGEX = re.compile(
    rf"""
    (?:(?P<city>[A-Za-zÁÉÍÓÚáéíóúÑñüÜ]+)\s*,\s*)?
    (?:(?P<weekday>{WD_ALT})\s*,?\s+)?
    {CORE_DATE}
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Rescate en cuerpo: día de semana OBLIGATORIO (coma opcional).
RESCUE_REGEX = re.compile(
    rf"""
    (?P<weekday>{WD_ALT})\s*,?\s+
    {CORE_DATE}
    """,
    re.IGNORECASE | re.VERBOSE,
)
BLOCK_QUINCENA_RE = re.compile(
    rf"(?P<year>\d{{4}})\s+(?P<mn>\d{{2}})\s+(?P<month>{MONTH_ALT})\s+(?P<d1>\d{{1,2}})-(?P<d2>\d{{1,2}})",
    re.IGNORECASE,
)
BLOCK_MES_RE = re.compile(
    rf"(?P<year>\d{{4}})\s+(?P<mn>\d{{2}})\s+(?P<month>{MONTH_ALT})\s*$",
    re.IGNORECASE,
)
YEAR_ANY_RE = re.compile(r"\b(20[0-2]\d)\b")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def normalize_header_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    for bad, good in {
        "0ctubre": "octubre", "setiernbre": "setiembre", "miercoies": "miercoles",
        "sabádo": "sabado", "deI ": "del ", " deI ": " del ", " dei ": " del ",
    }.items():
        text = text.replace(bad, good)
    return text


def normalize_body_text(text: str) -> str:
    """Fixes mínimos para el rescate en cuerpo (sin colapsar saltos de línea)."""
    for bad, good in {
        "0ctubre": "octubre", "setiernbre": "setiembre", "miercoies": "miercoles",
        "sabádo": "sabado", " deI ": " del ", " dei ": " del ",
    }.items():
        text = text.replace(bad, good)
    return text


def fix_ocr_num(token: str) -> str:
    return token.replace("I", "1").replace("l", "1").replace("O", "0").replace("o", "0")


def clamp_range(year: int, month: int, d1: int, d2: int) -> Tuple[date, date]:
    last = calendar.monthrange(year, month)[1]
    d1 = max(1, min(d1, last))
    d2 = max(1, min(d2, last))
    if d2 < d1:
        d1, d2 = d2, d1
    return date(year, month, d1), date(year, month, d2)


def block_range_from_parts(parts: Tuple[str, ...]) -> Tuple[Optional[Tuple[date, date]], str]:
    """Rango permitido desde las carpetas ancestro (la más profunda gana)."""
    for name in reversed(parts):
        m = BLOCK_QUINCENA_RE.search(name)
        if m:
            try:
                y = int(m.group("year")); mo = MONTHS[m.group("month").lower()]
                return clamp_range(y, mo, int(m.group("d1")), int(m.group("d2"))), "quincena"
            except (ValueError, KeyError):
                pass
        m = BLOCK_MES_RE.search(name)
        if m:
            try:
                y = int(m.group("year")); mo = MONTHS[m.group("month").lower()]
                last = calendar.monthrange(y, mo)[1]
                return (date(y, mo, 1), date(y, mo, last)), "mes"
            except (ValueError, KeyError):
                pass
    for name in reversed(parts):
        m = YEAR_ANY_RE.search(name)
        if m:
            y = int(m.group(1))
            return (date(y, 1, 1), date(y, 12, 31)), "anio"
    return None, "sin_bloque"


def candidate_score(dt: date, year: int, weekday: Optional[str],
                    wk_ok: Optional[bool], match_start: int,
                    raw_match: str, allowed: Optional[Tuple[date, date]],
                    header_lower: str) -> int:
    score = 0
    if weekday:
        score += 2
        if wk_ok is True:
            score += 8          # checksum verificado (1/7 de acertar por azar)
        elif wk_ok is False:
            score -= 12         # día de semana no cuadra: dígito mal leído
    if 2001 <= year <= 2022:
        score += 3
    if match_start <= 120:
        score += 3
    elif match_start <= 220:
        score += 1
    if allowed is not None:
        score += 10 if (allowed[0] <= dt <= allowed[1]) else -20
    if year == 1921:
        score -= 10
    if "fundado" in header_lower and match_start > header_lower.find("fundado") >= 0:
        score -= 8
    if "16 de septiembre de 1921" in raw_match.lower():
        score -= 15
    return score


def parse_match(m: re.Match) -> Optional[date]:
    try:
        day = int(fix_ocr_num(m.group("day")))
        year = int(fix_ocr_num(m.group("year")))
        month = MONTHS[m.group("month").lower()]
    except (ValueError, KeyError):
        return None
    if year < 1000:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_date(header_text: str,
                 allowed: Optional[Tuple[date, date]]) -> Optional[Dict[str, Any]]:
    header_lower = header_text.lower()
    candidates: List[Dict[str, Any]] = []
    for m in DATE_REGEX.finditer(header_text):
        dt = parse_match(m)
        if dt is None:
            continue
        wk = m.group("weekday").lower() if m.group("weekday") else None
        wk_ok = (dt.weekday() == WEEKDAYS[wk]) if wk in WEEKDAYS else None
        candidates.append({
            "dt": dt, "weekday_ok": wk_ok,
            "score": candidate_score(dt, dt.year, wk, wk_ok, m.start(),
                                     m.group(0), allowed, header_lower),
        })
    if not candidates:
        return None
    if allowed is not None:
        in_range = [c for c in candidates if allowed[0] <= c["dt"] <= allowed[1]]
        if not in_range:
            return None      # fuera del bloque: mejor sin fecha exacta que un 1921
        candidates = in_range
    return max(candidates, key=lambda c: c["score"])


def extract_body_anchor(body_text: str,
                        allowed: Optional[Tuple[date, date]]) -> Optional[date]:
    """Ancla rescatada del cuerpo: weekday obligatorio + consistente + en rango.
    Si la página vota por varias fechas, gana la más repetida (folio/pie suelen
    repetirse); empate -> la de posición más temprana."""
    if allowed is None:
        return None
    votes: Dict[date, int] = {}
    first_pos: Dict[date, int] = {}
    for m in RESCUE_REGEX.finditer(body_text):
        dt = parse_match(m)
        if dt is None:
            continue
        wk = m.group("weekday").lower()
        if wk not in WEEKDAYS or dt.weekday() != WEEKDAYS[wk]:
            continue
        if not (allowed[0] <= dt <= allowed[1]):
            continue
        votes[dt] = votes.get(dt, 0) + 1
        first_pos.setdefault(dt, m.start())
    if not votes:
        return None
    return max(votes, key=lambda d: (votes[d], -first_pos[d]))


# ── Limpieza (soft_clean_raw con fix del ÍNDICE) ────────────────────────────
MATH_TAG_RE = re.compile(r"</?math>", re.IGNORECASE)
GENERIC_TAG_RE = re.compile(r"</?(sup|sub|span|b|i|u|em|strong)>", re.IGNORECASE)
WEB_LINE_RE = re.compile(r"^\s*(DIRECCI[ÓO]N EN INTERNET:)?\s*www\.\S+\s*$", re.IGNORECASE)
IDX_START_RE = re.compile(r"^\s*##\s*[ÍI]NDICE\s*$", re.IGNORECASE)
IDX_END_RE = re.compile(r"^\s*##\s+\S")


def soft_clean_raw(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = html.unescape(md)
    md = MATH_TAG_RE.sub("", md)
    md = GENERIC_TAG_RE.sub("", md)
    md = HTML_COMMENT_RE.sub("", md)
    out, skipping = [], False
    for ln in md.splitlines():
        t = ln.rstrip()
        s = t.strip()
        if WEB_LINE_RE.match(s):
            continue
        if IDX_START_RE.match(s):
            skipping = True
            continue
        if skipping:
            if IDX_END_RE.match(s):
                skipping = False
                out.append(t)
            continue
        out.append(t)
    text = re.sub(r"\n{4,}", "\n\n\n", "\n".join(out))
    return text.strip() + "\n"


# ── Segmentación por encabezados ## ─────────────────────────────────────────
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")


def split_sections(text: str) -> List[Tuple[str, str]]:
    """[(titulo, texto), ...] — ## consecutivos se fusionan; preámbulo con titulo ''."""
    rows: List[Tuple[str, str]] = []
    title_parts: List[str] = []
    body: List[str] = []

    def body_text() -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()

    def flush() -> None:
        titulo = " — ".join(p for p in title_parts if p).strip()
        texto = body_text()
        if titulo or texto:
            rows.append((titulo, texto))

    for ln in text.splitlines():
        m = HEADING_RE.match(ln)
        if m:
            if body_text():
                flush()
                title_parts, body = [], []
            title_parts.append(m.group(1).strip())
        else:
            body.append(ln)
    flush()
    return rows


# ── Resolución de fechas por bloque (v3: anclas header + cuerpo) ───────────
def resolve_block_dates(pages: List[Dict[str, Any]], counters: Dict[str, int]) -> None:
    """Cascada: header > cuerpo (monótono) > adelante > atrás > inicio del bloque.

    v3.1 — MODO DISPERSO para bloques SIN ningún header: en esos lotes
    (típicamente reprocesados) el orden de las carpetas de página no es
    necesariamente cronológico, así que ni la monotonía ni la herencia
    entre páginas son válidas. Cada página con ancla de cuerpo va a su día
    exacto (weekday consistente + rango del bloque bastan) y las demás caen
    a bloque_*: honestas en lo mensual, fuera de lo diario. Con esto,
    cuerpo_ant/cuerpo_retro solo existen en bloques ordenados (con header),
    donde la herencia sí es confiable."""
    n = len(pages)
    for p in pages:
        p["anchor"] = "header" if p["fecha"] else ""

    if not any(p["anchor"] == "header" for p in pages):
        for p in pages:
            cdt: Optional[date] = p.get("cuerpo_dt")
            if cdt is not None:
                p["fecha"] = cdt.isoformat()
                p["fuente"] = "cuerpo"
            elif p["allowed"] is not None:
                p["fecha"] = p["allowed"][0].isoformat()
                p["fuente"] = f"bloque_{p['level']}"
            else:
                p["fuente"] = "sin_fecha"
            counters[p["fuente"]] = counters.get(p["fuente"], 0) + 1
        return

    # siguiente ancla de HEADER a partir de cada posición (para monotonía)
    next_hdr: List[Optional[date]] = [None] * n
    nh: Optional[date] = None
    for i in range(n - 1, -1, -1):
        if pages[i]["anchor"] == "header":
            nh = date.fromisoformat(pages[i]["fecha"])
        next_hdr[i] = nh

    # aceptar anclas de cuerpo que respeten la monotonía del orden de escaneo
    last_dt: Optional[date] = None
    for i, p in enumerate(pages):
        if p["anchor"] == "header":
            last_dt = date.fromisoformat(p["fecha"])
            continue
        cdt: Optional[date] = p.get("cuerpo_dt")
        if cdt is None:
            continue
        lo = last_dt if last_dt is not None else (p["allowed"][0] if p["allowed"] else None)
        hi = next_hdr[i] if next_hdr[i] is not None else (p["allowed"][1] if p["allowed"] else None)
        if lo is not None and cdt < lo:
            continue
        if hi is not None and cdt > hi:
            continue
        p["fecha"] = cdt.isoformat()
        p["anchor"] = "cuerpo"
        last_dt = cdt

    # propagación hacia adelante desde la última ancla (header o cuerpo)
    last: Optional[str] = None
    last_kind = ""
    for p in pages:
        if p["anchor"]:
            p["fuente"] = p["anchor"]
            last, last_kind = p["fecha"], p["anchor"]
        elif last is not None:
            p["fecha"] = last
            p["fuente"] = "pagina_anterior" if last_kind == "header" else "cuerpo_ant"

    # relleno hacia atrás desde la primera ancla
    nxt: Optional[str] = None
    nxt_kind = ""
    for p in reversed(pages):
        if p["anchor"]:
            nxt, nxt_kind = p["fecha"], p["anchor"]
        elif not p["fecha"] and nxt is not None:
            p["fecha"] = nxt
            p["fuente"] = "retro" if nxt_kind == "header" else "cuerpo_retro"

    # bloque sin ninguna ancla -> inicio del rango del bloque
    for p in pages:
        if not p["fecha"]:
            if p["allowed"] is not None:
                p["fecha"] = p["allowed"][0].isoformat()
                p["fuente"] = f"bloque_{p['level']}"
            else:
                p["fuente"] = "sin_fecha"
        counters[p["fuente"]] = counters.get(p["fuente"], 0) + 1


# ── Pipeline ────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default=DEF_IN_ROOT)
    ap.add_argument("--out-dir", default=DEF_OUT_DIR)
    args = ap.parse_args()
    in_root, out_dir = Path(args.in_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_CSV

    t0 = time.time()
    grand = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "periodico", "path", "fecha", "fecha_fuente",
                    "titulo", "texto"])

        for paper in PAPERS:
            root = in_root / paper
            if not root.is_dir():
                print(f"[skip] no existe: {root}", flush=True)
                continue
            n = empty = rows_out = 0
            counters: Dict[str, int] = {}
            cur_block = None
            block_pages: List[Dict[str, Any]] = []

            def flush_block() -> None:
                nonlocal rows_out, empty
                if not block_pages:
                    return
                resolve_block_dates(block_pages, counters)
                for p in block_pages:
                    if not p["secs"]:
                        empty += 1
                    for k, (titulo, texto) in enumerate(p["secs"], 1):
                        w.writerow([f"{p['page']}_{k}", paper, p["rel"],
                                    p["fecha"], p["fuente"], titulo, texto])
                        rows_out += 1
                block_pages.clear()

            for md_file in sorted(root.rglob("raw.md")):
                try:
                    raw = md_file.read_text(encoding="utf-8", errors="ignore")
                except OSError as e:
                    print(f"[warn] no pude leer {md_file}: {e}", flush=True)
                    continue
                rel = md_file.parent.relative_to(in_root)
                parts = rel.parts[:-1]
                block_key = md_file.parent.parent
                if block_key != cur_block:
                    flush_block()
                    cur_block = block_key

                allowed, level = block_range_from_parts(parts)
                plain = HTML_COMMENT_RE.sub("", html.unescape(raw))
                header = normalize_header_text(plain[:600])[:400]
                got = extract_date(header, allowed)
                cuerpo_dt: Optional[date] = None
                if got is None:
                    cuerpo_dt = extract_body_anchor(normalize_body_text(plain),
                                                    allowed)
                block_pages.append({
                    "page": md_file.parent.name,
                    "rel": str(rel),
                    "fecha": got["dt"].isoformat() if got else "",
                    "fuente": "",
                    "cuerpo_dt": cuerpo_dt,
                    "allowed": allowed,
                    "level": level,
                    "secs": split_sections(soft_clean_raw(raw)),
                })
                n += 1
                if n % 25000 == 0:
                    print(f"  {paper}: {n} páginas / ~{rows_out} filas...", flush=True)
            flush_block()

            grand += rows_out
            dated = sum(v for k, v in counters.items() if k != "sin_fecha")
            pct = (100.0 * dated / n) if n else 0.0
            det = " | ".join(f"{k} {v}" for k, v in sorted(counters.items()))
            print(f"[ok] {paper}: {n} páginas -> {rows_out} filas | {det} | "
                  f"{pct:.1f}% fechadas | {empty} páginas sin secciones", flush=True)

    print(f"\nTOTAL filas (secciones): {grand} | {time.time()-t0:.0f}s | OUT: {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())