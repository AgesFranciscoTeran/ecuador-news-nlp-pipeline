#!/usr/bin/env python3
"""
verificar_recall.py (v2) — Cuánta calidad de recuperación cuesta el índice ANN,
y con qué parámetros se recupera.

Mide dos cosas distintas, porque la base se va a usar de dos maneras:

  1) BÚSQUEDA GLOBAL (sin filtro de fecha): el caso duro. Barre varias
     combinaciones de nprobes/refine_factor contra la búsqueda exacta y
     reporta recall@k y latencia de cada una. Sirve para elegir los
     parámetros de consulta del producto con evidencia.

  2) BÚSQUEDA CON VENTANA TEMPORAL: el patrón real del clic en la línea de
     tiempo. Al filtrar por fecha, el conjunto candidato se reduce miles de
     veces, así que la búsqueda EXACTA suele ser viable — y exacta significa
     100% de recall, sin depender del índice. Este bloque lo cronometra.

Las consultas de prueba son vectores de la propia base (chunks al azar), así
que no hace falta modelo ni descargar nada.

Uso:
  python3 verificar_recall.py --db vectordb --n 30 --k 10
  python3 verificar_recall.py --db vectordb --n 30 --k 10 --dias 7
"""
import argparse
import random
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Set, Tuple

# (nprobes, refine_factor) — None = valores por defecto del motor
CONFIGS: List[Tuple[Optional[int], Optional[int]]] = [
    (None, None), (32, 10), (128, 20), (256, 50), (512, 50), (1024, 50),
]


def ids(filas: List[dict]) -> Set[str]:
    return {f["chunk_uid"] for f in filas}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="vectordb")
    ap.add_argument("--tabla", default="corpus")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dias", type=int, default=7,
                    help="ancho de la ventana temporal a cronometrar")
    ap.add_argument("--muestra", type=int, default=5000)
    args = ap.parse_args()

    import lancedb
    db = lancedb.connect(args.db)
    tbl = db.open_table(args.tabla)
    total = tbl.count_rows()
    print(f"tabla: {total} chunks")
    for i in tbl.list_indices():
        det = getattr(i, "index_details", {}) or {}
        print(f"índice: {getattr(i, 'index_type', '?')} | "
              f"compresión: {det.get('compression', 'ninguna')}")
    print()

    cab = tbl.head(min(args.muestra, total)).to_pylist()
    if not cab:
        print("tabla vacía"); return 1
    random.seed(7)
    muestras = random.sample(cab, min(args.n, len(cab)))
    consultas = [r["vector"] for r in muestras]

    # ── 1) búsqueda global ────────────────────────────────────────────────
    print("== Búsqueda global (sin filtro) ==")
    verdad: List[Set[str]] = []
    t_exact = 0.0
    for v in consultas:
        t0 = time.perf_counter()
        c = (tbl.search(v, vector_column_name="vector")
             .bypass_vector_index().limit(args.k).to_list())
        t_exact += time.perf_counter() - t0
        verdad.append(ids(c))
    n = len(consultas)

    for nprobes, refine in CONFIGS:
        rec = 0.0
        t = 0.0
        for v, real in zip(consultas, verdad):
            q = tbl.search(v, vector_column_name="vector")
            if nprobes is not None:
                q = q.nprobes(nprobes)
            if refine is not None:
                q = q.refine_factor(refine)
            t0 = time.perf_counter()
            r = q.limit(args.k).to_list()
            t += time.perf_counter() - t0
            if real:
                rec += len(ids(r) & real) / len(real)
        etiqueta = ("por defecto" if nprobes is None
                    else f"nprobes={nprobes} refine={refine}")
        print(f"  {etiqueta:<28} recall@{args.k} {100*rec/n:5.1f}%   "
              f"{1000*t/n:8.1f} ms")
    print(f"  {'exacta (referencia)':<28} recall@{args.k} 100.0%   "
          f"{1000*t_exact/n:8.1f} ms")

    # ── 2) ventana temporal: el patrón del producto ───────────────────────
    print(f"\n== Ventana temporal de ±{args.dias} días (exacta, recall 100%) ==")
    t_win = 0.0
    n_res = 0
    usadas = 0
    for r, v in zip(muestras, consultas):
        f = r.get("fecha", "")
        if len(f) != 10:
            continue
        try:
            d = date.fromisoformat(f)
        except ValueError:
            continue
        d0 = (d - timedelta(days=args.dias)).isoformat()
        d1 = (d + timedelta(days=args.dias)).isoformat()
        t0 = time.perf_counter()
        res = (tbl.search(v, vector_column_name="vector")
               .where(f"fecha >= '{d0}' AND fecha <= '{d1}'", prefilter=True)
               .bypass_vector_index().limit(args.k).to_list())
        t_win += time.perf_counter() - t0
        n_res += len(res)
        usadas += 1
    if usadas:
        print(f"  {usadas} consultas | {1000*t_win/usadas:8.1f} ms | "
              f"{n_res/usadas:.1f} resultados de media")
        print("\n  Si esto responde en decenas o pocos cientos de ms, el clic en\n"
              "  la línea de tiempo puede ir SIEMPRE por búsqueda exacta: 100%\n"
              "  de recall garantizado y el índice queda solo para la búsqueda\n"
              "  global del corpus completo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())