#!/usr/bin/env python3
"""
consultar_vectordb.py — Búsqueda semántica sobre la base vectorial, con filtro
por fecha, periódico y origen. Es a la vez la herramienta de verificación del
entregable y la primitiva del producto final: el clic en un punto de la línea
de tiempo se traduce en una consulta con --desde/--hasta.

POLÍTICA DE BÚSQUEDA (medida sobre 9.484.560 chunks, índice IVF_SQ):

  * CON filtro de fechas -> búsqueda EXACTA (bypass del índice vectorial).
    El índice escalar BTree sobre `fecha` reduce el candidato a unos pocos
    miles de chunks, así que la exacta responde en ~59 ms con 100% de recall.
    Es el caso del clic en la línea de tiempo.
  * SIN filtro -> índice IVF_SQ afinado: nprobes=128 / refine_factor=20 da
    98,3% de recall@10 en ~383 ms; 256/50 sube a 99,7% en ~753 ms. Con los
    valores por defecto del motor caería a 83,7%, de ahí que se fijen aquí.
  * --ann fuerza el índice aunque haya filtro (útil para comparar).

COLAPSO POR ARTÍCULO: una nota larga se troceó en varios chunks y puede copar
el top-k con pedazos de sí misma. Por defecto se pide de más y se conserva el
mejor chunk de cada artículo, de modo que --k 10 devuelve 10 notas distintas.
Con --por-chunk se ve el ranking crudo.

El modelo con el que se construyó la base manda: `config.json` guarda modelo,
prefijos y (si aplica) el endpoint HTTP, y consultar con otro modelo se
rechaza en vez de devolver resultados sin sentido.

Ejemplos:
  # qué se decía en la ventana de la quiebra de Lehman
  python3 consultar_vectordb.py --db vectordb \\
      --q "crisis económica y precio del petróleo" \\
      --desde 2008-09-15 --hasta 2008-09-30 --k 10

  # sin filtro temporal, todo el corpus
  python3 consultar_vectordb.py --db vectordb --q "paro nacional y subsidios"

  # verificación rápida del entregable (sin modelo, no requiere GPU)
  python3 consultar_vectordb.py --db vectordb --modelo dummy --q "prueba"
"""
import argparse
import hashlib
import sys
from typing import Any, Dict, List


def vector_dummy(texto: str, dim: int) -> List[float]:
    h = hashlib.sha256(texto.encode("utf-8")).digest()
    v = [(h[i % len(h)] - 127.5) / 127.5 for i in range(dim)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def vector_http(cfg: dict, q: str, timeout: int) -> List[float]:
    import json as _json
    import urllib.request
    base = cfg["endpoint"].rstrip("/")
    texto = cfg.get("prefijo_query", "") + q
    proto = cfg.get("protocolo", "openai")
    if proto == "tei":
        path, payload = "/embed", {"inputs": [texto], "normalize": True,
                                   "truncate": True}
    elif proto == "ollama":
        path, payload = "/api/embed", {"model": cfg.get("modelo_servido"),
                                       "input": [texto]}
    else:
        path, payload = "/v1/embeddings", {"model": cfg.get("modelo_servido"),
                                           "input": [texto]}
    req = urllib.request.Request(
        base + path, data=_json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = _json.loads(r.read().decode("utf-8"))
    if proto == "tei":
        v = d[0]
    elif proto == "ollama":
        v = d["embeddings"][0]
    else:
        v = d["data"][0]["embedding"]
    v = [float(x) for x in v]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def colapsar_por_articulo(filas: List[Dict[str, Any]],
                          k: int) -> List[Dict[str, Any]]:
    """Un solo chunk por artículo. Las filas vienen ordenadas por distancia,
    así que el primero que se conserva de cada artículo es el mejor."""
    vistos, unicas = set(), []
    for r in filas:
        clave = r.get("id_maestro") or r.get("path") or r.get("chunk_uid")
        if clave in vistos:
            continue
        vistos.add(clave)
        unicas.append(r)
        if len(unicas) >= k:
            break
    return unicas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="vectordb")
    ap.add_argument("--tabla", default="corpus")
    ap.add_argument("--q", required=True, help="consulta en lenguaje natural")
    ap.add_argument("--modelo", default="",
                    help="por defecto se lee de config.json de la base")
    ap.add_argument("--desde", default="")
    ap.add_argument("--hasta", default="")
    ap.add_argument("--periodico", default="")
    ap.add_argument("--origen", default="", choices=["", "ocr", "webscraping"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chars", type=int, default=220)
    ap.add_argument("--nprobes", type=int, default=128)
    ap.add_argument("--refine", type=int, default=20)
    ap.add_argument("--ann", action="store_true",
                    help="usar el índice aun con filtro de fechas")
    ap.add_argument("--por-chunk", action="store_true", dest="por_chunk",
                    help="no colapsar: mostrar el ranking crudo de chunks")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--local", action="store_true",
                    help="ignorar el endpoint y cargar el modelo localmente")
    args = ap.parse_args()

    import json
    from pathlib import Path
    import lancedb

    # El modelo con el que se construyó la base manda: consultar con otro
    # devuelve resultados sin sentido y sin ningún error visible.
    cfg = {}
    cfg_path = Path(args.db) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    modelo = args.modelo or cfg.get("modelo") or "intfloat/multilingual-e5-large"
    if args.modelo and cfg.get("modelo") and args.modelo != cfg["modelo"]:
        print(f"[ERROR] la base fue construida con '{cfg['modelo']}'; "
              f"consultarla con '{args.modelo}' devuelve basura.", file=sys.stderr)
        return 2

    db = lancedb.connect(args.db)
    tbl = db.open_table(cfg.get("tabla", args.tabla))
    dim = tbl.schema.field("vector").type.list_size

    if cfg.get("endpoint") and not args.local:
        # el mismo servicio HTTP que se usó para indexar
        try:
            qv = vector_http(cfg, args.q, args.timeout)
        except Exception as e:                                     # noqa: BLE001
            print(f"[ERROR] no respondió el endpoint {cfg['endpoint']}: {e}\n"
                  f"        Los datos están intactos. Levanta el servicio, o "
                  f"consulta con el modelo local:\n"
                  f"        ... --local --modelo {cfg.get('modelo_servido') or cfg.get('modelo')}",
                  file=sys.stderr)
            return 3
    elif modelo == "dummy":
        qv = vector_dummy(args.q, dim)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(modelo)
        pref = cfg.get("prefijo_query",
                       "query: " if "e5" in modelo.lower() else "")
        qv = m.encode([pref + args.q], normalize_embeddings=True)[0].tolist()

    cond: List[str] = []
    if args.desde:
        cond.append(f"fecha >= '{args.desde}'")
    if args.hasta:
        cond.append(f"fecha <= '{args.hasta}'")
    if args.periodico:
        cond.append(f"periodico = '{args.periodico}'")
    if args.origen:
        cond.append(f"origen = '{args.origen}'")

    exacta = bool(cond) and not args.ann
    q = tbl.search(qv, vector_column_name="vector")
    if cond:
        q = q.where(" AND ".join(cond), prefilter=True)
    if exacta:
        q = q.bypass_vector_index()            # 100% recall, ~59 ms con BTree
    else:
        q = q.nprobes(args.nprobes).refine_factor(args.refine)

    # Se pide de más para poder quedarse con un solo chunk por artículo.
    pedidos = args.k if args.por_chunk else args.k * 4
    filas = q.limit(pedidos).to_list()
    n_chunks = len(filas)
    if not args.por_chunk:
        filas = colapsar_por_articulo(filas, args.k)

    if not filas:
        print("(sin resultados: revisa el rango de fechas o el filtro)")
        return 0

    modo = ("exacta (100% recall)" if exacta
            else f"índice nprobes={args.nprobes} refine={args.refine}")
    cab = f"consulta: {args.q!r}"
    if cond:
        cab += f" | ventana {args.desde or '…'} → {args.hasta or '…'}"
    cab += f" | modo: {modo} | {len(filas)} resultados"
    if not args.por_chunk:
        cab += f" (de {n_chunks} chunks)"
    print(cab + "\n")

    for i, r in enumerate(filas, 1):
        d = r.get("_distance")
        txt = " ".join((r.get("texto") or "").split())[:args.chars]
        print(f"{i:>2}. {r['fecha']}  {r['periodico']} ({r['origen']})"
              + (f"  d={d:.4f}" if d is not None else ""))
        if r.get("titulo"):
            print(f"    « {r['titulo'][:110]} »")
        print(f"    {txt}…")
        print(f"    [{r['fecha_fuente']}] {r.get('path') or r.get('id_maestro')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())