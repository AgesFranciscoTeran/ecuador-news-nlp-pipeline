#!/usr/bin/env python3
"""
vistazo_maestro.py (v2) — Enseña cómo luce el corpus maestro sin abrir los 9,7 GB.

Tres usos:

  1) FICHAS — imprime filas completas en formato legible, campo por campo, en
     vez de líneas de CSV ilegibles. Sirve para mostrar el esquema en pantalla.

  2) LOCALIZAR EN EL CSV lo que devolvió la base vectorial. Cada resultado de
     `consultar_vectordb.py` trae su `path` (y su `id_maestro`), así que con
     --path / --id se recupera la fila original con su texto completo. Se
     pueden pasar VARIOS criterios en una sola corrida, que es lo que conviene:
     el CSV se recorre una sola vez y se resuelven todos de golpe.

  3) CENSO (--censo) — resumen de la composición del corpus: filas por origen,
     periódico, confianza de fechado, flag, y un histograma por año.

Nota de tiempo: el maestro tiene primero las filas del webscraping y después
las del OCR, así que buscar filas del OCR recorre varios millones de líneas.
Cuenta unos minutos y corre todo lo que necesites en una sola pasada.

Ejemplos:
  # las tres fichas de contraste para la presentación
  python3 vistazo_maestro.py --maestro M --origen webscraping --n 1
  python3 vistazo_maestro.py --maestro M --flags clasificado --n 1

  # las mismas notas que devolvió la búsqueda vectorial, con texto completo
  python3 vistazo_maestro.py --maestro M --completo \\
      --path "2008 09 septiembre/00603" \\
      --path "2008 09 septiembre 16-30/00651" \\
      --path "2008 09 septiembre 16-30/00617"

  # por texto del titular
  python3 vistazo_maestro.py --maestro M --contiene "El petróleo sigue su carrera" --completo

  python3 vistazo_maestro.py --maestro M --censo > censo_maestro.txt
"""
import argparse
import csv
import signal
import sys
import unicodedata
from collections import Counter

# permite canalizar la salida a `head` sin que Python lance BrokenPipeError
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):
    pass

csv.field_size_limit(sys.maxsize)

CAMPOS = ["id", "origen", "id_original", "periodico", "fecha", "fecha_fuente",
          "seccion", "titulo", "texto", "flags", "path"]


def recorta(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def ficha(r: dict, ancho: int, completo: bool) -> str:
    out = ["─" * 78]
    for c in CAMPOS:
        v = r.get(c, "") or ""
        if c == "texto":
            v = " ".join(v.split()) if completo else recorta(v, ancho)
            if completo:
                out.append(f"  {c:<13} ({len(v)} caracteres)")
                out.append("")
                for i in range(0, len(v), 76):
                    out.append(f"    {v[i:i+76]}")
                continue
        elif c == "titulo":
            v = recorta(v, 110)
        else:
            v = recorta(v, 90)
        out.append(f"  {c:<13} {v if v else '(vacío)'}")
    return "\n".join(out)


def coincide(r: dict, a) -> bool:
    if a.origen and r.get("origen") != a.origen:
        return False
    if a.periodico and a.periodico.lower() not in (r.get("periodico") or "").lower():
        return False
    if a.fecha and r.get("fecha") != a.fecha:
        return False
    if a.id and r.get("id") not in a.id:
        return False
    if a.path and not any(p in (r.get("path") or "") for p in a.path):
        return False
    if a.contiene:
        campo = sin_tildes((r.get("titulo") or "") + " " + (r.get("texto") or ""))
        if not all(sin_tildes(t) in campo for t in a.contiene):
            return False
    if a.flags:
        fl = {f for f in (r.get("flags") or "").split(";") if f}
        if a.flags == "ninguno":
            if fl:
                return False
        elif a.flags not in fl:
            return False
    if a.min_texto and len(r.get("texto") or "") < a.min_texto:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", required=True)
    ap.add_argument("--n", type=int, default=3, help="máximo de fichas")
    ap.add_argument("--ancho", type=int, default=400, help="caracteres de texto")
    ap.add_argument("--completo", action="store_true",
                    help="imprimir el texto entero, no un extracto")
    ap.add_argument("--origen", default="", choices=["", "ocr", "webscraping"])
    ap.add_argument("--periodico", default="")
    ap.add_argument("--fecha", default="", help="AAAA-MM-DD exacta")
    ap.add_argument("--id", action="append", default=[],
                    help="id exacto del maestro (repetible)")
    ap.add_argument("--path", action="append", default=[],
                    help="fragmento del path, como el que imprime la búsqueda (repetible)")
    ap.add_argument("--contiene", action="append", default=[],
                    help="texto que debe aparecer en título o cuerpo (repetible)")
    ap.add_argument("--flags", default="",
                    help="clasificado | numerico | tabla | solo_titulo | ninguno")
    ap.add_argument("--min-texto", type=int, default=0, dest="min_texto")
    ap.add_argument("--censo", action="store_true")
    ap.add_argument("--limite", type=int, default=0, help="tope de filas leídas")
    a = ap.parse_args()

    if not a.censo:
        vistas = 0
        with open(a.maestro, encoding="utf-8-sig", newline="") as fh:
            rd = csv.DictReader(fh)
            print(f"columnas: {', '.join(rd.fieldnames or [])}\n")
            for i, r in enumerate(rd, 1):
                if a.limite and i > a.limite:
                    break
                if coincide(r, a):
                    print(ficha(r, a.ancho, a.completo))
                    vistas += 1
                    if vistas >= a.n:
                        break
                if i % 2000000 == 0:
                    print(f"  ... {i:,} filas revisadas", file=sys.stderr, flush=True)
        print("─" * 78)
        if not vistas:
            print("(ninguna fila coincide con el filtro)")
        return 0

    # ── Censo ──────────────────────────────────────────────────────────────
    n = 0
    por_origen, por_periodico, por_fuente, por_flag = (Counter() for _ in range(4))
    por_anio = Counter()
    sin_flag = 0
    chars = 0
    fmin, fmax = "9999", "0000"
    with open(a.maestro, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            n += 1
            por_origen[r.get("origen", "?")] += 1
            por_periodico[r.get("periodico", "?")] += 1
            por_fuente[r.get("fecha_fuente", "?")] += 1
            f = r.get("fecha", "")
            if len(f) == 10:
                por_anio[f[:4]] += 1
                fmin = min(fmin, f)
                fmax = max(fmax, f)
            fl = [x for x in (r.get("flags") or "").split(";") if x]
            if fl:
                for x in fl:
                    por_flag[x] += 1
            else:
                sin_flag += 1
            chars += len(r.get("texto") or "")
            if n % 1000000 == 0:
                print(f"  ... {n:,} filas", file=sys.stderr, flush=True)

    def bloque(titulo: str, c: Counter, total: int) -> None:
        print(f"\n{titulo}")
        for k, v in c.most_common():
            print(f"  {k:<22} {v:>10,}  {100*v/total:5.1f}%")

    print("=" * 60)
    print(f"CORPUS MAESTRO — {n:,} filas")
    print(f"rango de fechas: {fmin} → {fmax}")
    print(f"texto total: {chars/1e9:.1f} mil millones de caracteres "
          f"({chars/n:.0f} por fila de media)")
    print("=" * 60)
    bloque("Por origen", por_origen, n)
    bloque("Por periódico", por_periodico, n)
    bloque("Por confianza de fechado (fecha_fuente)", por_fuente, n)
    print("\nPor flag (una fila puede llevar varios)")
    print(f"  {'sin flag (artículos)':<22} {sin_flag:>10,}  {100*sin_flag/n:5.1f}%")
    for k, v in por_flag.most_common():
        print(f"  {k:<22} {v:>10,}  {100*v/n:5.1f}%")
    print("\nFilas por año")
    tope = max(por_anio.values()) if por_anio else 1
    for k in sorted(por_anio):
        v = por_anio[k]
        print(f"  {k}  {v:>9,}  {'█' * max(1, round(40 * v / tope))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())