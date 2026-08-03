#!/usr/bin/env python3
"""
auditar_calidad.py (v2) — Auditoría comparativa de calidad textual entre dos
corpus CSV, en streaming.

CAMBIO CLAVE respecto a v1 (los scores NO son comparables entre versiones):
La v1 contaba los números como "palabras inválidas", de modo que el score
mezclaba dos cosas distintas: (a) calidad de extracción del texto y
(b) composición del contenido (clasificados, cotizaciones, horarios — llenos
de cifras legítimas). La v2 las separa:

  * % palabras válidas (alfabéticas): de los tokens que CONTIENEN letras,
    cuántos son palabras plausibles en español. Mide extracción, no contenido.
    El score se basa en esto.
  * % tokens numéricos puros: cuota de cifras (teléfonos, precios, horas).
    Mide composición; NO penaliza el score.
  * mezcla sospechosa /100 pal: tokens letra+dígito EXCLUYENDO abreviaturas
    legítimas del español (1er, 2do, 3ra, 16h30, m2, km2...). Lo que queda
    sí huele a OCR ("c0mercio", "Quit0").

Score por fila (0–100):
    score = 100 * (válidas / tokens alfabéticos)
            − 2·(mezcla sospechosa /100 pal)
            − 1.5·(mayúsculas internas /100 pal)
            − 3·(chars raros /100 chars)
            − 2·(cortes de guion /100 pal)
Filas sin ningún token alfabético puntúan 0 (son tablas/listados puros; se
identifican mejor con los flags del limpiador).

Exporta por corpus: peores_<origen>.csv (K peores filas) y
muestras_<origen>.csv (N filas al azar, semilla fija).

Uso:
  python3 auditar_calidad.py --viejo articulos.csv --nuevo corpus.csv \
      --out-dir informe_calidad [--peores 200] [--muestras 30] [--max-filas 0]
"""
import argparse
import csv
import heapq
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

csv.field_size_limit(sys.maxsize)

# ── Repertorio de caracteres esperable en prensa en español ────────────────
PERMITIDOS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "ÁÉÍÓÚÜÑáéíóúüñ0123456789"
    " \t\n\r.,;:¡!¿?()[]{}\"'`´«»“”‘’%$€#@&+*=_/\\<>~^°ºª©®™·—–\u00ad…\u00a0-"
)
VOCALES = set("aáeéiíoóuúü")
CONSONANTES_RE = re.compile(r"[bcdfghjklmnñpqrstvwxyz]{5,}", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")
MAYUS_INTERNA_RE = re.compile(r"[a-záéíóúüñ][A-ZÁÉÍÓÚÜÑ]")
CORTE_GUION_RE = re.compile(r"[a-záéíóúüñ]-\s+[a-záéíóúüñ]")
TAG_HTML_RE = re.compile(r"</?[a-zA-Z][^>\n]{0,30}>")
ENTIDAD_RE = re.compile(r"&[a-zA-Z#][a-zA-Z0-9]{1,8};")
MONOSILABAS_1 = {"a", "e", "o", "u", "y"}

# Abreviaturas legítimas letra+dígito del español de prensa:
# ordinales (1er, 2do, 3ra, 4to...), horas (16h30, 09H00), unidades (m2, km2).
LEGIT_MIX_RE = re.compile(
    r"^(?:\d{1,3}(?:er|ero|era|do|da|ro|ra|to|ta|mo|ma|vo|va|no|na)s?"
    r"|\d{1,2}[hH]\d{2}"
    r"|(?:m|km|cm|mm)[23]"
    r"|\d+(?:m|km|kg|g|cc|cm|mm|ha|m2|m3|km2))$",
    re.IGNORECASE,
)


def es_palabra_valida(tok: str) -> bool:
    if not tok.isalpha():
        return False
    if len(tok) == 1:
        return tok.lower() in MONOSILABAS_1
    if len(tok) > 24:
        return False
    if tok.isupper() and 2 <= len(tok) <= 6:      # siglas: OK
        return True
    low = tok.lower()
    if not any(c in VOCALES for c in low):
        return False
    if CONSONANTES_RE.search(low):
        return False
    return True


def analizar_texto(texto: str) -> Dict[str, Any]:
    n_chars = len(texto)
    alfa = dig = punt = raros = 0
    for ch in texto:
        if ch.isalpha():
            alfa += 1
        elif ch.isdigit():
            dig += 1
        elif ch in PERMITIDOS:
            if not ch.isspace():
                punt += 1
        else:
            raros += 1

    tokens = TOKEN_RE.findall(texto)
    n_tok = len(tokens)
    tok_alfa = tok_num = validas = mezcla = may_int = uni_malas = 0
    for t in tokens:
        tiene_l = any(c.isalpha() for c in t)
        if not tiene_l:
            tok_num += 1
            continue
        tok_alfa += 1
        if any(c.isdigit() for c in t):
            if LEGIT_MIX_RE.match(t):
                validas += 1          # 1er, 2do, 16h30, m2: español legítimo
            else:
                mezcla += 1
            continue
        if MAYUS_INTERNA_RE.search(t):
            may_int += 1
        if es_palabra_valida(t):
            validas += 1
        elif len(t) == 1:
            uni_malas += 1

    cortes = len(CORTE_GUION_RE.findall(texto))
    tags = len(TAG_HTML_RE.findall(texto))
    entidades = len(ENTIDAD_RE.findall(texto))
    almohadillas = texto.count("##")
    pipes = texto.count("|")

    if tok_alfa == 0:
        score = 0.0
    else:
        p100 = 100.0 / tok_alfa
        c100 = (100.0 / n_chars) if n_chars else 0.0
        score = (100.0 * validas / tok_alfa
                 - 2.0 * mezcla * p100
                 - 1.5 * may_int * p100
                 - 3.0 * raros * c100
                 - 2.0 * cortes * p100)
        score = max(0.0, min(100.0, score))

    return {
        "chars": n_chars, "alfa": alfa, "dig": dig, "punt": punt, "raros": raros,
        "tokens": n_tok, "tok_alfa": tok_alfa, "tok_num": tok_num,
        "validas": validas, "mezcla": mezcla, "may_int": may_int,
        "uni_malas": uni_malas, "cortes": cortes, "tags": tags,
        "entidades": entidades, "almohadillas": almohadillas,
        "pipes": pipes, "score": score,
    }


class Acumulador:
    def __init__(self, origen: str, k_peores: int, n_muestras: int) -> None:
        self.origen = origen
        self.filas = 0
        self.txt_vacios = 0
        self.tit_vacios = 0
        self.longitudes: List[int] = []
        self.tot: Dict[str, int] = {}
        self.hist = [0] * 101
        self.k = k_peores
        self.peores: List[Any] = []
        self.n_m = n_muestras
        self.muestras: List[List[str]] = []
        self._rnd = random.Random(42)
        self._seq = 0

    def agregar(self, fila_id: str, titulo: str, texto: str) -> None:
        self.filas += 1
        if not titulo.strip():
            self.tit_vacios += 1
        if not texto.strip():
            self.txt_vacios += 1
        m = analizar_texto(texto)
        self.longitudes.append(m["chars"])
        for k in ("chars", "alfa", "dig", "punt", "raros", "tokens", "tok_alfa",
                  "tok_num", "validas", "mezcla", "may_int", "uni_malas",
                  "cortes", "tags", "entidades", "almohadillas", "pipes"):
            self.tot[k] = self.tot.get(k, 0) + m[k]
        sc = m["score"]
        self.hist[int(round(sc))] += 1

        self._seq += 1
        extracto = re.sub(r"\s+", " ", texto)[:200]
        item = (-sc, self._seq, fila_id, round(sc, 1), extracto)
        if len(self.peores) < self.k:
            heapq.heappush(self.peores, item)
        elif -sc > self.peores[0][0]:
            heapq.heapreplace(self.peores, item)

        if len(self.muestras) < self.n_m:
            self.muestras.append([fila_id, round(sc, 1), extracto])
        else:
            j = self._rnd.randint(1, self.filas)
            if j <= self.n_m:
                self.muestras[j - 1] = [fila_id, round(sc, 1), extracto]

    def pct(self, num: str, den: str) -> float:
        d = self.tot.get(den, 0)
        return 100.0 * self.tot.get(num, 0) / d if d else 0.0

    def por_100(self, k: str, den: str = "tok_alfa") -> float:
        d = self.tot.get(den, 0)
        return 100.0 * self.tot.get(k, 0) / d if d else 0.0

    def percentil(self, p: float) -> float:
        total = sum(self.hist)
        if not total:
            return 0.0
        objetivo = p / 100.0 * total
        acum = 0
        for s, c in enumerate(self.hist):
            acum += c
            if acum >= objetivo:
                return float(s)
        return 100.0

    def resumen(self) -> Dict[str, float]:
        med = statistics.median(self.longitudes) if self.longitudes else 0
        prom = statistics.fmean(self.longitudes) if self.longitudes else 0
        return {
            "filas": self.filas,
            "textos vacíos": self.txt_vacios,
            "títulos vacíos": self.tit_vacios,
            "longitud media (chars)": round(prom, 1),
            "longitud mediana (chars)": med,
            "% chars alfabéticos": round(self.pct("alfa", "chars"), 2),
            "% chars dígitos": round(self.pct("dig", "chars"), 2),
            "% chars raros": round(self.pct("raros", "chars"), 4),
            "% tokens numéricos puros": round(self.pct("tok_num", "tokens"), 2),
            "% palabras válidas (alfabéticas)": round(self.pct("validas", "tok_alfa"), 2),
            "mezcla sospechosa /100 pal": round(self.por_100("mezcla"), 3),
            "mayúscula interna /100 pal": round(self.por_100("may_int"), 3),
            "1-letra inválidas /100 pal": round(self.por_100("uni_malas"), 3),
            "cortes de guion /100 pal": round(self.por_100("cortes"), 3),
            "tags HTML (total)": self.tot.get("tags", 0),
            "entidades &..; (total)": self.tot.get("entidades", 0),
            "'##' markdown (total)": self.tot.get("almohadillas", 0),
            "pipes '|' (total)": self.tot.get("pipes", 0),
            "score p5": self.percentil(5),
            "score p25": self.percentil(25),
            "score mediana": self.percentil(50),
            "score p75": self.percentil(75),
        }


def lower_keys(row: Dict[str, str]) -> Dict[str, str]:
    return {(k or "").strip().lower(): (v or "") for k, v in row.items()}


def procesar(path: Path, origen: str, k_peores: int, n_muestras: int,
             max_filas: int) -> Acumulador:
    ac = Acumulador(origen, k_peores, n_muestras)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for i, raw in enumerate(csv.DictReader(fh), 1):
            r = lower_keys(raw)
            fila_id = r.get("id", "") or str(i)
            ac.agregar(fila_id, r.get("titulo", ""), r.get("texto", ""))
            if max_filas and i >= max_filas:
                break
    return ac


def escribir_csv(path: Path, header: List[str], filas: List[List[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(filas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viejo", required=True, help="CSV de webscraping")
    ap.add_argument("--nuevo", required=True, help="corpus.csv del OCR")
    ap.add_argument("--out-dir", default="informe_calidad")
    ap.add_argument("--peores", type=int, default=200)
    ap.add_argument("--muestras", type=int, default=30)
    ap.add_argument("--max-filas", type=int, default=0,
                    help="límite de filas por corpus (0 = todo; útil para probar)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpora = [
        ("webscraping", Path(args.viejo)),
        ("ocr", Path(args.nuevo)),
    ]
    resultados: List[Acumulador] = []
    for origen, path in corpora:
        print(f"[..] analizando {origen}: {path}", flush=True)
        ac = procesar(path, origen, args.peores, args.muestras, args.max_filas)
        resultados.append(ac)
        peores = sorted(ac.peores, key=lambda t: -t[0])
        escribir_csv(out / f"peores_{origen}.csv",
                     ["id", "score", "extracto"],
                     [[t[2], t[3], t[4]] for t in peores])
        escribir_csv(out / f"muestras_{origen}.csv",
                     ["id", "score", "extracto"], ac.muestras)

    a, b = resultados[0].resumen(), resultados[1].resumen()
    ancho = max(len(k) for k in a) + 2
    lineas = [f"{'MÉTRICA (auditor v2)':<{ancho}}{'webscraping':>16}{'ocr':>16}"]
    lineas.append("-" * (ancho + 32))
    for k in a:
        lineas.append(f"{k:<{ancho}}{a[k]:>16}{b[k]:>16}")
    lineas.append("-" * (ancho + 32))
    delta = a["score mediana"] - b["score mediana"]
    lineas.append(f"Brecha de score mediano (webscraping − ocr): {delta:+.0f} puntos")
    informe = "\n".join(lineas)
    print("\n" + informe)
    (out / "informe.txt").write_text(informe + "\n", encoding="utf-8")
    print(f"\n[ok] Informe y CSVs de inspección en: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
