#!/usr/bin/env python3
"""
construir_vectordb.py — Base de datos vectorial del corpus maestro (LanceDB).

Pensada para dos cosas a la vez:
  (a) ser el entregable: un DIRECTORIO de archivos que se copia tal cual al
      disco duro y se abre en cualquier máquina sin levantar servidores;
  (b) soportar la ACTUALIZACIÓN MENSUAL del compañero de webscraping sin
      re-embeber los 9M de filas cada vez.

Cómo logra (b): manifiesto por (periodico, origen, mes). En cada corrida
cuenta cuántas filas indexables hay en cada mes y las compara con lo que ya
está en la base. Los meses idénticos se saltan; los que cambiaron de conteo
(mes nuevo o corregido) se borran de la tabla y se reindexan. Es decir: se
vuelve a correr el MISMO comando cada mes y solo se procesa lo nuevo.

Qué entra al índice (filtros por defecto, todos configurables):
  * capa de fechado `extendida` (excluye bloque_* : fechas de confianza
    mensual no sirven para recuperar "qué pasó el 3 de enero de 2008");
  * filas tipo artículo: se excluyen los flags tabla / numerico / clasificado
    (siguen en el maestro y en las señales; no son texto embebible);
  * mínimo de caracteres (--min-chars, default 200): descarta fragmentos que
    solo ensucian el top-k.

Chunking: cada fila larga se parte en trozos de --max-chars con --solape,
cortando en frontera de oración cuando se puede. El TÍTULO se antepone a cada
chunk (un trozo del medio de la nota sin titular es casi irrecuperable).

Backend de embeddings intercambiable:
  --modelo intfloat/multilingual-e5-large   (default; prefijos "passage:"/"query:")
  --modelo BAAI/bge-m3                      (alternativa, contexto largo)
  --modelo dummy                            (SIN modelo: vectores deterministas
                                             de prueba, para validar el pipeline
                                             completo en segundos)

Uso:
  # ensayo en seco: valida filtros, chunking, manifiesto y escritura
  python3 construir_vectordb.py --maestro csv_corpus/corpus_maestro.csv \
      --db vectordb --modelo dummy --limite 200000

  # construcción real
  python3 construir_vectordb.py --maestro csv_corpus/corpus_maestro.csv \
      --db vectordb --modelo intfloat/multilingual-e5-large --indexar

  # actualización mensual (mismo comando; solo procesa lo nuevo)
  python3 construir_vectordb.py --maestro csv_corpus/corpus_maestro.csv \
      --db vectordb --modelo intfloat/multilingual-e5-large --indexar
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

csv.field_size_limit(sys.maxsize)

CAPAS = {
    "estricta": {"web", "header", "pagina_anterior", "retro"},
    "extendida": {"web", "header", "pagina_anterior", "retro",
                  "cuerpo", "cuerpo_ant", "cuerpo_retro"},
}
FLAGS_EXCLUIR = {"tabla", "numerico", "clasificado"}
TABLA = "corpus"
FIN_ORACION_RE = re.compile(r"(?<=[.!?])\s+")


# ── Embeddings ──────────────────────────────────────────────────────────────
class DummyEmbedder:
    """Vectores deterministas sin modelo: prueba el pipeline de punta a punta."""
    prefijo_doc = ""
    prefijo_query = ""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, textos: List[str]) -> List[List[float]]:
        out = []
        for t in textos:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = [(h[i % len(h)] - 127.5) / 127.5 for i in range(self.dim)]
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


class STEmbedder:
    """sentence-transformers en GPU (fp16). Normaliza a norma 1 -> coseno."""

    def __init__(self, modelo: str, batch: int, device: Optional[str]) -> None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = SentenceTransformer(modelo, device=dev)
        if dev.startswith("cuda"):
            self.m = self.m.half()
        self.batch = batch
        self.dim = self.m.get_sentence_embedding_dimension()
        es_e5 = "e5" in modelo.lower()
        self.prefijo_doc = "passage: " if es_e5 else ""
        self.prefijo_query = "query: " if es_e5 else ""
        print(f"[..] modelo {modelo} | dim {self.dim} | device {dev} | "
              f"prefijo doc {self.prefijo_doc!r}", flush=True)

    def encode(self, textos: List[str]) -> List[List[float]]:
        v = self.m.encode(textos, batch_size=self.batch,
                          normalize_embeddings=True, show_progress_bar=False,
                          convert_to_numpy=True)
        return v.astype("float32").tolist()


class HTTPEmbedder:
    """Endpoint de embeddings servido por HTTP en la propia H200.

    Detecta solo el protocolo, porque no siempre se sabe qué está levantado:
      * TEI (HuggingFace text-embeddings-inference):  GET /info   + POST /embed
      * compatible OpenAI (vLLM, Infinity, LocalAI):  GET /v1/models + POST /v1/embeddings
      * Ollama:                                       GET /api/tags + POST /api/embed

    Guardarraíl importante: registra el ID del modelo que el servidor dice
    estar sirviendo. Como el endpoint lo controla otra persona, puede cambiar
    de modelo sin avisar — y consultar con otro modelo devuelve basura sin
    error. Ese ID queda en config.json y se verifica en cada corrida.
    """

    def __init__(self, endpoint: str, modelo: str, batch: int, timeout: int,
                 reintentos: int, paralelo: int) -> None:
        self.base = endpoint.rstrip("/")
        self.batch = batch
        self.timeout = timeout
        self.reintentos = reintentos
        self.paralelo = max(1, paralelo)
        self.protocolo, self.modelo_id = self._detectar(modelo)
        prueba = self._encode_lote(["prueba de dimensión"])
        self.dim = len(prueba[0])
        es_e5 = "e5" in (self.modelo_id or modelo).lower()
        self.prefijo_doc = "passage: " if es_e5 else ""
        self.prefijo_query = "query: " if es_e5 else ""
        print(f"[..] endpoint {self.base} | protocolo {self.protocolo} | "
              f"modelo servido {self.modelo_id!r} | dim {self.dim} | "
              f"{self.paralelo} peticiones en paralelo", flush=True)

    # ── transporte ─────────────────────────────────────────────────────────
    def _pedir(self, path: str, payload: Optional[dict]):
        datos = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=datos,
            method="POST" if datos is not None else "GET",
            headers={"Content-Type": "application/json"})
        ultimo = None
        for intento in range(self.reintentos):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:                                # noqa: BLE001
                ultimo = e
                if intento + 1 < self.reintentos:
                    time.sleep(min(2 ** intento, 15))
        raise RuntimeError(f"{path}: {ultimo}")

    def _detectar(self, modelo: str):
        try:                                    # TEI
            info = self._pedir("/info", None)
            if isinstance(info, dict) and info.get("model_id"):
                return "tei", info["model_id"]
        except Exception:                                          # noqa: BLE001
            pass
        try:                                    # compatible OpenAI
            ms = self._pedir("/v1/models", None)
            ids = [d.get("id") for d in (ms or {}).get("data", []) if d.get("id")]
            if ids:
                return "openai", (modelo if modelo in ids else ids[0])
        except Exception:                                          # noqa: BLE001
            pass
        try:                                    # Ollama
            tags = self._pedir("/api/tags", None)
            nombres = [m.get("name") for m in (tags or {}).get("models", [])]
            if nombres:
                return "ollama", (modelo if modelo in nombres else nombres[0])
        except Exception:                                          # noqa: BLE001
            pass
        print("[warn] no pude identificar el protocolo del endpoint; asumo "
              "compatible con OpenAI (/v1/embeddings)", flush=True)
        return "openai", modelo

    # ── embeddings ─────────────────────────────────────────────────────────
    @staticmethod
    def _norma(v: List[float]) -> List[float]:
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def _encode_lote(self, textos: List[str]) -> List[List[float]]:
        if self.protocolo == "tei":
            r = self._pedir("/embed", {"inputs": textos, "normalize": True,
                                       "truncate": True})
            vecs = r
        elif self.protocolo == "ollama":
            r = self._pedir("/api/embed", {"model": self.modelo_id,
                                           "input": textos})
            vecs = r["embeddings"]
        else:
            r = self._pedir("/v1/embeddings", {"model": self.modelo_id,
                                               "input": textos})
            datos = sorted(r["data"], key=lambda d: d.get("index", 0))
            vecs = [d["embedding"] for d in datos]
        if len(vecs) != len(textos):
            raise RuntimeError(f"el endpoint devolvió {len(vecs)} vectores "
                               f"para {len(textos)} textos")
        return [self._norma([float(x) for x in v]) for v in vecs]

    def encode(self, textos: List[str]) -> List[List[float]]:
        lotes = [textos[i:i + self.batch]
                 for i in range(0, len(textos), self.batch)]
        if self.paralelo == 1 or len(lotes) == 1:
            out: List[List[float]] = []
            for l in lotes:
                out.extend(self._encode_lote(l))
            return out
        with ThreadPoolExecutor(max_workers=self.paralelo) as ex:
            partes = list(ex.map(self._encode_lote, lotes))
        return [v for p in partes for v in p]


# ── Lectura y troceado ──────────────────────────────────────────────────────
def trocear(texto: str, max_chars: int, solape: int) -> List[str]:
    if len(texto) <= max_chars:
        return [texto]
    partes: List[str] = []
    ini = 0
    while ini < len(texto):
        fin = min(ini + max_chars, len(texto))
        if fin < len(texto):
            ventana = texto[ini:fin]
            cortes = [m.end() for m in FIN_ORACION_RE.finditer(ventana)]
            if cortes and cortes[-1] > max_chars * 0.5:
                fin = ini + cortes[-1]
        partes.append(texto[ini:fin].strip())
        if fin >= len(texto):
            break
        ini = max(fin - solape, ini + 1)
    return [p for p in partes if p]


def fila_indexable(r: Dict[str, str], capa: Optional[Set[str]],
                   min_chars: int, desde: str, hasta: str) -> bool:
    if capa is not None and r.get("fecha_fuente", "") not in capa:
        return False
    fecha = r.get("fecha", "")
    if len(fecha) != 10:
        return False
    if desde and fecha < desde:
        return False
    if hasta and fecha > hasta:
        return False
    flags = {f for f in (r.get("flags", "") or "").split(";") if f}
    if flags & FLAGS_EXCLUIR:
        return False
    largo = len(r.get("titulo", "") or "") + len(r.get("texto", "") or "")
    return largo >= min_chars


def leer(maestro: Path, capa: Optional[Set[str]], min_chars: int,
         desde: str, hasta: str, limite: int) -> Iterator[Dict[str, str]]:
    with maestro.open("r", encoding="utf-8-sig", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), 1):
            if limite and i > limite:
                break
            if fila_indexable(r, capa, min_chars, desde, hasta):
                yield r


def clave_mes(r: Dict[str, str]) -> Tuple[str, str, str]:
    return (r.get("periodico", "?"), r.get("origen", "?"), r["fecha"][:7])


# ── Base vectorial ──────────────────────────────────────────────────────────
def esquema(dim: int):
    import pyarrow as pa
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("chunk_uid", pa.string()),
        pa.field("id_maestro", pa.string()),
        pa.field("origen", pa.string()),
        pa.field("periodico", pa.string()),
        pa.field("fecha", pa.string()),
        pa.field("mes", pa.string()),
        pa.field("anio", pa.int32()),
        pa.field("fecha_fuente", pa.string()),
        pa.field("seccion", pa.string()),
        pa.field("titulo", pa.string()),
        pa.field("texto", pa.string()),
        pa.field("path", pa.string()),
        pa.field("chunk_idx", pa.int32()),
        pa.field("n_chunks", pa.int32()),
    ])


def esc(s: str) -> str:
    return s.replace("'", "''")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", required=True)
    ap.add_argument("--db", default="vectordb")
    ap.add_argument("--modelo", default="intfloat/multilingual-e5-large")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--endpoint", default="",
                    help="URL de un servidor de embeddings en la H200 "
                         "(TEI, compatible OpenAI u Ollama). Si se pasa, se usa "
                         "en vez de cargar el modelo localmente.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--reintentos", type=int, default=4)
    ap.add_argument("--paralelo", type=int, default=4,
                    help="peticiones HTTP simultáneas contra el endpoint")
    ap.add_argument("--capa", choices=["estricta", "extendida", "todo"],
                    default="extendida")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--solape", type=int, default=150)
    ap.add_argument("--desde", default="")
    ap.add_argument("--hasta", default="")
    ap.add_argument("--limite", type=int, default=0,
                    help="tope de filas leídas del maestro (pruebas)")
    ap.add_argument("--rehacer", action="store_true",
                    help="reindexar también los meses ya presentes")
    ap.add_argument("--indexar", action="store_true",
                    help="crear el índice ANN al final (hazlo en la carga masiva)")
    args = ap.parse_args()

    import lancedb

    maestro = Path(args.maestro)
    dbdir = Path(args.db)
    dbdir.mkdir(parents=True, exist_ok=True)
    manif_path = dbdir / "manifiesto.json"
    capa: Optional[Set[str]] = None if args.capa == "todo" else CAPAS[args.capa]
    t0 = time.time()

    manifiesto: Dict[str, int] = {}
    if manif_path.exists():
        manifiesto = json.loads(manif_path.read_text(encoding="utf-8"))
        print(f"[..] manifiesto: {len(manifiesto)} meses-medio ya indexados",
              flush=True)

    # ── Contrato de la base: el modelo queda fijado y se verifica ─────────
    # El motor de base de datos se puede cambiar reimportando vectores; el
    # MODELO no: indexar con uno y consultar con otro devuelve basura sin dar
    # ningún error. Se comprueba antes de hacer cualquier trabajo.
    cfg_path = dbdir / "config.json"
    cfg_prev = (json.loads(cfg_path.read_text(encoding="utf-8"))
                if cfg_path.exists() else None)
    if cfg_prev and cfg_prev.get("modelo") != args.modelo:
        print(f"[ERROR] esta base fue construida con '{cfg_prev.get('modelo')}' "
              f"y estás pasando '{args.modelo}'.\n"
              f"        Mezclar embeddings de dos modelos rompe la búsqueda en "
              f"silencio.\n"
              f"        Para cambiar de modelo, construye en un --db nuevo.",
              file=sys.stderr)
        return 2

    # ── Pasada 1: conteo por mes-medio y decisión de qué (re)indexar ───────
    print("[..] pasada 1/2: censo de meses", flush=True)
    conteo: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in leer(maestro, capa, args.min_chars, args.desde, args.hasta,
                  args.limite):
        conteo[clave_mes(r)] += 1

    objetivo: Set[Tuple[str, str, str]] = set()
    nuevos = cambiados = saltados = 0
    for k, n in conteo.items():
        key = "|".join(k)
        prev = manifiesto.get(key)
        if prev is None:
            objetivo.add(k)
            nuevos += 1
        elif prev != n or args.rehacer:
            objetivo.add(k)
            cambiados += 1
        else:
            saltados += 1
    print(f"     {len(conteo)} meses-medio | nuevos {nuevos} | "
          f"cambiados {cambiados} | sin cambios {saltados}", flush=True)
    if not objetivo:
        print("[ok] nada que hacer: la base ya está al día.")
        return 0

    # ── Modelo ─────────────────────────────────────────────────────────────
    if args.endpoint:
        emb = HTTPEmbedder(args.endpoint, args.modelo, args.batch,
                           args.timeout, args.reintentos, args.paralelo)
    elif args.modelo == "dummy":
        emb = DummyEmbedder()
    else:
        emb = STEmbedder(args.modelo, args.batch, args.device)

    servido = getattr(emb, "modelo_id", None)
    if cfg_prev and servido and cfg_prev.get("modelo_servido") not in (None, servido):
        print(f"[ERROR] el endpoint ahora sirve '{servido}' y la base se "
              f"construyó con '{cfg_prev.get('modelo_servido')}'.\n"
              f"        Alguien cambió el modelo del servicio: los vectores "
              f"nuevos no serían comparables con los ya indexados.",
              file=sys.stderr)
        return 2
    if cfg_prev and cfg_prev.get("dim") != emb.dim:
        print(f"[ERROR] dimensión {emb.dim} != {cfg_prev.get('dim')} registrada.",
              file=sys.stderr)
        return 2
    cfg = {
        "modelo": args.modelo, "dim": emb.dim,
        "endpoint": args.endpoint or None,
        "protocolo": getattr(emb, "protocolo", None),
        "modelo_servido": servido,
        "prefijo_doc": emb.prefijo_doc, "prefijo_query": emb.prefijo_query,
        "capa": args.capa, "min_chars": args.min_chars,
        "max_chars": args.max_chars, "solape": args.solape,
        "flags_excluidos": sorted(FLAGS_EXCLUIR), "tabla": TABLA,
        "actualizado": time.strftime("%Y-%m-%d %H:%M"),
    }
    if cfg_prev:
        difs = [k for k in ("capa", "min_chars", "max_chars", "solape")
                if cfg_prev.get(k) != cfg[k]]
        if difs:
            print(f"[warn] cambian parámetros de troceado ({', '.join(difs)}): "
                  f"los meses nuevos quedarán con criterios distintos a los "
                  f"ya indexados", flush=True)
        cfg["creado"] = cfg_prev.get("creado", cfg["actualizado"])
    else:
        cfg["creado"] = cfg["actualizado"]
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    db = lancedb.connect(str(dbdir))
    try:                       # abrir/crear sin depender de la API de listado,
        tbl = db.open_table(TABLA)          # que cambia entre versiones
    except Exception:                                          # noqa: BLE001
        tbl = db.create_table(TABLA, schema=esquema(emb.dim))

    # borrar meses que se van a reindexar (corrección o --rehacer)
    for (per, ori, mes) in objetivo:
        if manifiesto.get("|".join((per, ori, mes))) is not None:
            tbl.delete(f"periodico = '{esc(per)}' AND origen = '{esc(ori)}' "
                       f"AND mes = '{esc(mes)}'")

    # ── Pasada 2: chunking + embeddings + carga ────────────────────────────
    print("[..] pasada 2/2: embeddings", flush=True)
    buf_txt: List[str] = []
    buf_meta: List[Dict] = []
    n_filas = n_chunks = 0

    def volcar() -> None:
        nonlocal buf_txt, buf_meta
        if not buf_txt:
            return
        vecs = emb.encode(buf_txt)
        for m, v in zip(buf_meta, vecs):
            m["vector"] = v
        tbl.add(buf_meta)
        buf_txt, buf_meta = [], []

    for r in leer(maestro, capa, args.min_chars, args.desde, args.hasta,
                  args.limite):
        k = clave_mes(r)
        if k not in objetivo:
            continue
        titulo = (r.get("titulo", "") or "").strip()
        cuerpo = (r.get("texto", "") or "").strip()
        partes = trocear(cuerpo, args.max_chars, args.solape) or [""]
        n_filas += 1
        for j, parte in enumerate(partes):
            visible = (titulo + ". " + parte).strip(" .") if titulo else parte
            if len(visible) < args.min_chars and len(partes) > 1:
                continue
            uid = f"{r.get('origen','')}:{r.get('id','')}:{j}"
            buf_txt.append(emb.prefijo_doc + visible)
            buf_meta.append({
                "chunk_uid": uid, "id_maestro": r.get("id", ""),
                "origen": r.get("origen", ""), "periodico": k[0],
                "fecha": r["fecha"], "mes": k[2], "anio": int(r["fecha"][:4]),
                "fecha_fuente": r.get("fecha_fuente", ""),
                "seccion": r.get("seccion", "") or "",
                "titulo": titulo, "texto": visible,
                "path": r.get("path", "") or "",
                "chunk_idx": j, "n_chunks": len(partes),
            })
            n_chunks += 1
            if len(buf_txt) >= args.batch * 4:
                volcar()
                if n_chunks % 100000 < args.batch * 4:
                    print(f"     {n_chunks} chunks | {time.time()-t0:.0f}s",
                          flush=True)
    volcar()

    for k, n in conteo.items():
        if k in objetivo:
            manifiesto["|".join(k)] = n
    manif_path.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=0,
                                     sort_keys=True), encoding="utf-8")

    if args.indexar and n_chunks > 10000:
        print("[..] creando índice ANN (puede tardar)", flush=True)
        try:
            tbl.create_index(metric="cosine", vector_column_name="vector",
                             replace=True)
            print("     índice creado", flush=True)
        except Exception as e:                     # noqa: BLE001
            print(f"[warn] no se pudo crear el índice: {e}\n"
                  "       la búsqueda funciona igual (escaneo exacto, más lento)",
                  flush=True)

    print(f"\n[ok] {n_filas} filas -> {n_chunks} chunks indexados | "
          f"{time.time()-t0:.0f}s")
    print(f"     tabla '{TABLA}': {tbl.count_rows()} chunks totales")
    print(f"     manifiesto: {len(manifiesto)} meses-medio | DB: {dbdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())