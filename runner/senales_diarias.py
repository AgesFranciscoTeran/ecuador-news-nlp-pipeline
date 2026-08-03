#!/usr/bin/env python3
"""
senales_diarias.py — Convierte el corpus maestro en un PANEL DIARIO de señales
para el nowcasting. Es el puente entre "tengo 9M de filas de texto" y "tengo
una serie de alta frecuencia que puedo graficar y calibrar contra el PIB".

Decisiones de diseño (explícitas, para poder defenderlas):

1) MEDIO = (periodico, origen). El Universo impreso (~344 filas/día) y El
   Universo web (~61 artículos/día) son regímenes de extracción distintos:
   tratarlos como un solo medio crearía un salto de nivel en el costurón
   2019-2022. Se estandarizan por separado y se promedian como dos medios.

2) SEÑAL EN PROPORCIONES, NUNCA EN CONTEOS. share_econ = artículos económicos
   / artículos del día. Así un día con menos páginas escaneadas o la salida de
   un diario del panel no se confunde con un shock económico.

3) CAPA DE FECHADO configurable con --capa:
     estricta  = web, header, pagina_anterior, retro
     extendida = estricta + cuerpo, cuerpo_ant, cuerpo_retro   (default)
     todo      = incluye bloque_* (solo tiene sentido para agregados mensuales)

4) FLUJOS SEPARADOS. Los artículos alimentan share_econ / tono / EPU; los
   flags del limpiador se cuentan aparte (clasificados = proxy tipo
   Help-Wanted Index, tablas y numéricos = cotizaciones). No se mezclan.

5) LÉXICO v0, calibrable. ECON / POS / NEG / INCERT / POLICY son listas de
   raíces (sin tildes, ya normalizadas). El índice tipo EPU exige las tres
   familias en el mismo artículo (económico + política + incertidumbre), al
   estilo Baker-Bloom-Davis. El sentimiento por diccionario es un punto de
   partida grueso: "aumento" es positivo salvo en "aumento del desempleo".
   Por eso el script también VALIDA el léxico económico contra la columna
   `seccion` del webscraping (que trae la sección real del diario) y reporta
   precisión/recall — con eso se afina la lista antes de creerle al índice.

Salidas en --out-dir:
  senales_diarias.csv — panel largo: una fila por (fecha, periodico, origen)
  indice_diario.csv   — índice combinado: z-scores por medio promediados por
                        día, con medias móviles de 7 y 30 observaciones
  lexico_vs_seccion.txt — validación del léxico contra las secciones web

Uso:
  python3 senales_diarias.py --maestro csv_corpus/corpus_maestro.csv \
      --out-dir senales [--capa extendida] [--min-art 10] [--max-filas 0]
"""
import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

csv.field_size_limit(sys.maxsize)

# ── Normalización rápida (translate es C, no unicodedata por fila) ─────────
TRANS = str.maketrans("áéíóúüñ", "aeiouun")
MAX_CHARS = 3000          # tope de análisis por fila (mediana real ~300)

CAPAS = {
    "estricta": {"web", "header", "pagina_anterior", "retro"},
    "extendida": {"web", "header", "pagina_anterior", "retro",
                  "cuerpo", "cuerpo_ant", "cuerpo_retro"},
}
FLAGS_NO_ARTICULO = {"tabla", "numerico", "clasificado"}

# ── Léxicos v0 (raíces, sin tildes) ────────────────────────────────────────
ECON = [
    r"econom\w*", r"\bpib\b", r"inflaci\w*", r"deflaci\w*", r"desemple\w*",
    r"emple[oa]\w*", r"salari\w*", r"sueldo\w*", r"\bsbu\b", r"canasta basica",
    r"export\w*", r"import\w*", r"arancel\w*", r"balanza comercial",
    r"comercio exterior", r"precio\w*", r"tarifa\w*", r"costo de vida",
    r"petrole\w*", r"crudo\b", r"barril\w*", r"\bopep\b", r"refineri\w*",
    r"gasolin\w*", r"diesel\b", r"combustible\w*", r"subsidi\w*",
    r"dolarizaci\w*", r"divisa\w*", r"remesa\w*", r"banc[oa]\w*",
    r"credit\w*", r"prestam\w*", r"tasa de interes", r"iliquidez",
    r"inversi[oó]n\w*", r"empresari\w*", r"\bempresas?\b", r"industri\w*",
    r"manufactur\w*", r"producci[oó]n\w*", r"productiv\w*", r"\bventas\b",
    r"tributari\w*", r"impuest\w*", r"\bsri\b", r"\biva\b", r"fisc[ao]\w*",
    r"presupuest\w*", r"deuda\w*", r"deficit\w*", r"superavit\w*",
    r"agricol\w*", r"agricultur\w*", r"banano\w*", r"camaron\w*", r"cacao\b",
    r"floricol\w*", r"pesquer\w*", r"miner[ioa]\w*", r"construcci[oó]n\w*",
    r"cemento\b", r"vivienda\w*", r"hipotec\w*", r"turism\w*", r"hoteler\w*",
    r"aerolinea\w*", r"transporte de carga", r"\bpuerto\w*", r"aduana\w*",
    r"\biess\b", r"jubilaci\w*", r"pension\w*", r"pobreza\b", r"\bfmi\b",
    r"banco mundial", r"\bcepal\b", r"riesgo pais", r"\bbonos?\b",
]
NEG = [
    r"crisis\b", r"recesi\w*", r"desaceleraci\w*", r"contracci[oó]n\w*",
    r"caida\w*", r"desplome\w*", r"derrumbe\w*", r"quiebra\w*", r"insolven\w*",
    r"morosidad\w*", r"impago\w*", r"despid[oa]\w*", r"desemple\w*",
    r"paraliza\w*", r"\bparo\b", r"huelga\w*", r"protesta\w*", r"perdida\w*",
    r"austerid\w*", r"recorte\w*", r"devaluaci\w*", r"escasez\b",
    r"desabastec\w*", r"apagon\w*", r"racionamiento\w*", r"congelamiento\w*",
    r"feriado bancario", r"incumpl\w*", r"deterior\w*", r"retroces\w*",
]
POS = [
    r"crecimiento\w*", r"creci[oó]\b", r"crece\b", r"recuperaci\w*",
    r"repunte\w*", r"expansi[oó]n\w*", r"aument[oa]\w*", r"increment\w*",
    r"mejor[oa]\w*", r"\bauge\b", r"dinamism\w*", r"record\b", r"superavit\w*",
    r"utilidad\w*", r"ganancia\w*", r"rentab\w*", r"reactivaci\w*",
    r"contrataci\w*", r"acuerdo comercial", r"impuls[oa]\w*",
]
INCERT = [
    r"incertidumbre\w*", r"inciert\w*", r"\briesgo\w*", r"inestabil\w*",
    r"volatil\w*", r"imprevis\w*", r"\bduda\w*", r"temor\w*", r"preocupa\w*",
    r"amenaza\w*", r"expectativa\w*",
]
POLICY = [
    r"gobierno\w*", r"presidente\b", r"asamblea\b", r"ministr\w*", r"\bley\b",
    r"leyes\b", r"decreto\w*", r"reforma\w*", r"regulaci\w*", r"politica economica",
    r"consulta popular", r"referend\w*", r"elecci[oó]n\w*", r"elecciones\b",
    r"banco central", r"superintendencia\w*", r"contraloria\b", r"\bfmi\b",
]


def compilar(lst: List[str]) -> re.Pattern:
    return re.compile("|".join(lst))


ECON_RE, NEG_RE, POS_RE = compilar(ECON), compilar(NEG), compilar(POS)
INCERT_RE, POLICY_RE = compilar(INCERT), compilar(POLICY)
SECCION_ECON_RE = re.compile(r"econom|negocio|dinero|mercado|finanz")


class Celda:
    __slots__ = ("n_total", "n_art", "n_econ", "n_epu", "tok_pos", "tok_neg",
                 "n_art_neg", "n_clasif", "n_tabla", "n_num", "n_solotit",
                 "chars_art")

    def __init__(self) -> None:
        self.n_total = self.n_art = self.n_econ = self.n_epu = 0
        self.tok_pos = self.tok_neg = self.n_art_neg = 0
        self.n_clasif = self.n_tabla = self.n_num = self.n_solotit = 0
        self.chars_art = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", required=True)
    ap.add_argument("--out-dir", default="senales")
    ap.add_argument("--capa", choices=["estricta", "extendida", "todo"],
                    default="extendida")
    ap.add_argument("--min-art", type=int, default=10,
                    help="mínimo de artículos por día-medio para entrar al índice")
    ap.add_argument("--max-filas", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    capa: Optional[Set[str]] = None if args.capa == "todo" else CAPAS[args.capa]

    panel: Dict[Tuple[str, str, str], Celda] = defaultdict(Celda)
    # validación del léxico contra la sección real (solo filas web)
    vp = vn = fp = fn = 0
    n_leidas = n_fuera_capa = 0

    with Path(args.maestro).open("r", encoding="utf-8-sig", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), 1):
            n_leidas = i
            ff = r.get("fecha_fuente", "")
            if capa is not None and ff not in capa:
                n_fuera_capa += 1
                if args.max_filas and i >= args.max_filas:
                    break
                continue
            fecha = r.get("fecha", "")
            if len(fecha) != 10:
                if args.max_filas and i >= args.max_filas:
                    break
                continue
            c = panel[(fecha, r.get("periodico", "?"), r.get("origen", "?"))]
            c.n_total += 1

            flags = {f for f in (r.get("flags", "") or "").split(";") if f}
            if "clasificado" in flags:
                c.n_clasif += 1
            if "tabla" in flags:
                c.n_tabla += 1
            if "numerico" in flags:
                c.n_num += 1
            if "solo_titulo" in flags:
                c.n_solotit += 1
            if flags & FLAGS_NO_ARTICULO:
                if args.max_filas and i >= args.max_filas:
                    break
                continue

            titulo = r.get("titulo", "") or ""
            texto = r.get("texto", "") or ""
            t = (titulo + " " + texto)[:MAX_CHARS].lower().translate(TRANS)
            if len(t) < 40:
                if args.max_filas and i >= args.max_filas:
                    break
                continue
            c.n_art += 1
            c.chars_art += len(t)

            es_econ = ECON_RE.search(t) is not None
            if es_econ:
                c.n_econ += 1
                npos = len(POS_RE.findall(t))
                nneg = len(NEG_RE.findall(t))
                c.tok_pos += npos
                c.tok_neg += nneg
                if nneg > npos:
                    c.n_art_neg += 1
                if INCERT_RE.search(t) and POLICY_RE.search(t):
                    c.n_epu += 1

            if r.get("origen") == "webscraping":
                sec = (r.get("seccion", "") or "").lower().translate(TRANS)
                if sec:
                    real = SECCION_ECON_RE.search(sec) is not None
                    if real and es_econ:
                        vp += 1
                    elif real and not es_econ:
                        fn += 1
                    elif not real and es_econ:
                        fp += 1
                    else:
                        vn += 1

            if i % 1000000 == 0:
                print(f"  {i} filas...", flush=True)
            if args.max_filas and i >= args.max_filas:
                break

    # ── Panel largo ────────────────────────────────────────────────────────
    def ratio(a: int, b: int) -> str:
        return f"{a / b:.6f}" if b else ""

    filas_panel = 0
    with (out / "senales_diarias.csv").open("w", newline="",
                                            encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["fecha", "periodico", "origen", "n_total", "n_art",
                    "n_econ", "share_econ", "n_epu", "share_epu",
                    "tok_pos", "tok_neg", "tono", "share_art_neg",
                    "n_clasif", "n_tabla", "n_num", "n_solotit", "chars_art"])
        for (fecha, per, ori), c in sorted(panel.items()):
            tono = ""
            if c.tok_pos + c.tok_neg:
                tono = f"{(c.tok_pos - c.tok_neg) / (c.tok_pos + c.tok_neg):.6f}"
            w.writerow([fecha, per, ori, c.n_total, c.n_art, c.n_econ,
                        ratio(c.n_econ, c.n_art), c.n_epu,
                        ratio(c.n_epu, c.n_art), c.tok_pos, c.tok_neg, tono,
                        ratio(c.n_art_neg, c.n_econ), c.n_clasif, c.n_tabla,
                        c.n_num, c.n_solotit, c.chars_art])
            filas_panel += 1

    # ── Índice combinado: z por medio, promedio de medios disponibles ──────
    METRICAS = ("share_econ", "tono", "share_epu")
    vals: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {m: [] for m in METRICAS})
    porfecha: Dict[str, List[Tuple[Tuple[str, str], Dict[str, float]]]] = defaultdict(list)

    for (fecha, per, ori), c in panel.items():
        if c.n_art < args.min_art:
            continue
        d: Dict[str, float] = {"share_econ": c.n_econ / c.n_art,
                               "share_epu": c.n_epu / c.n_art}
        if c.tok_pos + c.tok_neg:
            d["tono"] = (c.tok_pos - c.tok_neg) / (c.tok_pos + c.tok_neg)
        medio = (per, ori)
        for m, v in d.items():
            vals[medio][m].append(v)
        porfecha[fecha].append((medio, d))

    stats: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for medio, mv in vals.items():
        stats[medio] = {}
        for m, serie in mv.items():
            if len(serie) >= 30:
                mu = statistics.fmean(serie)
                sd = statistics.pstdev(serie)
                if sd > 1e-9:
                    stats[medio][m] = (mu, sd)

    fechas = sorted(porfecha)
    serie_idx: List[Tuple[str, int, Dict[str, Optional[float]]]] = []
    for fecha in fechas:
        acc: Dict[str, List[float]] = {m: [] for m in METRICAS}
        medios = set()
        for medio, d in porfecha[fecha]:
            st = stats.get(medio, {})
            for m, v in d.items():
                if m in st:
                    mu, sd = st[m]
                    acc[m].append((v - mu) / sd)
                    medios.add(medio)
        serie_idx.append((fecha, len(medios),
                          {m: (statistics.fmean(acc[m]) if acc[m] else None)
                           for m in METRICAS}))

    def movil(serie: List[Optional[float]], k: int) -> List[Optional[float]]:
        out_: List[Optional[float]] = []
        buf: List[float] = []
        for v in serie:
            if v is not None:
                buf.append(v)
                if len(buf) > k:
                    buf.pop(0)
            out_.append(statistics.fmean(buf) if buf else None)
        return out_

    cols = {m: [d[m] for _, _, d in serie_idx] for m in METRICAS}
    ma7 = {m: movil(cols[m], 7) for m in METRICAS}
    ma30 = {m: movil(cols[m], 30) for m in METRICAS}

    def fmt(v: Optional[float]) -> str:
        return f"{v:.6f}" if v is not None else ""

    with (out / "indice_diario.csv").open("w", newline="",
                                          encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["fecha", "n_medios"]
                   + [f"z_{m}" for m in METRICAS]
                   + [f"ma7_{m}" for m in METRICAS]
                   + [f"ma30_{m}" for m in METRICAS])
        for j, (fecha, nm, d) in enumerate(serie_idx):
            w.writerow([fecha, nm]
                       + [fmt(d[m]) for m in METRICAS]
                       + [fmt(ma7[m][j]) for m in METRICAS]
                       + [fmt(ma30[m][j]) for m in METRICAS])

    # ── Validación del léxico contra la sección real (filas web) ──────────
    L = ["== Léxico económico vs. sección real del webscraping ==",
         f"  verdaderos positivos: {vp}", f"  falsos positivos:    {fp}",
         f"  falsos negativos:    {fn}", f"  verdaderos negativos: {vn}"]
    if vp + fp:
        L.append(f"  precisión: {vp / (vp + fp):.3f}")
    if vp + fn:
        L.append(f"  recall:    {vp / (vp + fn):.3f}")
    L.append("\nNota: la 'verdad' es la sección del diario (Economía/Negocios/"
             "Mercados), que no es exactamente 'artículo con contenido "
             "económico': una nota de Política sobre el presupuesto es "
             "económica y cuenta como falso positivo. Sirve para calibrar el "
             "léxico, no como veredicto.")
    txt = "\n".join(L)
    (out / "lexico_vs_seccion.txt").write_text(txt + "\n", encoding="utf-8")

    print(f"\n[ok] {n_leidas} filas leídas | {n_fuera_capa} fuera de la capa "
          f"'{args.capa}'")
    print(f"     panel: {filas_panel} celdas (fecha × medio) -> "
          f"{out / 'senales_diarias.csv'}")
    print(f"     índice: {len(serie_idx)} días con al menos un medio "
          f"(min-art {args.min_art}) -> {out / 'indice_diario.csv'}")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
