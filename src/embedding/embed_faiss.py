from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import requests
import faiss

def ollama_embed(texts, model: str, host: str):
    # Ollama embeddings endpoint
    url = host.rstrip("/") + "/api/embeddings"
    vecs = []
    for t in texts:
        r = requests.post(url, json={"model": model, "prompt": t}, timeout=120)
        r.raise_for_status()
        v = r.json()["embedding"]
        vecs.append(v)
    return np.array(vecs, dtype="float32")

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="JSONL chunks (HQ o ALL)")
    ap.add_argument("--outdir", required=True, help="Directorio salida índice")
    ap.add_argument("--model", default="nomic-embed-text", help="Modelo de embeddings en Ollama")
    ap.add_argument("--host", default="http://localhost:11434", help="Host Ollama")
    ap.add_argument("--batch", type=int, default=16, help="Batch size (secuencia, no paralelo)")
    ap.add_argument("--normalize", action="store_true", help="Normalizar (cosine vía inner product)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path = outdir / "meta.jsonl"
    index_path = outdir / "faiss.index"

    # leer todo (puedes hacerlo streaming, pero para 30k chunks está ok)
    rows = list(iter_jsonl(inp))
    print("chunks:", len(rows))

    # embeddar
    all_vecs = []
    with meta_path.open("w", encoding="utf-8") as mf:
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i+args.batch]
            texts = [x["text"] for x in batch]
            vecs = ollama_embed(texts, model=args.model, host=args.host)

            if args.normalize:
                faiss.normalize_L2(vecs)

            all_vecs.append(vecs)

            # guardar metadata en el mismo orden que los vectores
            for x in batch:
                rec = {
                    "id": f'{x["source_path"]}::#{x["chunk_index"]}',
                    "source_path": x["source_path"],
                    "chunk_index": x["chunk_index"],
                    "quality_score": x.get("quality_score", None),
                    "text": x["text"],
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i // args.batch) % 20 == 0:
                print(f"embeddings: {min(i+args.batch, len(rows))}/{len(rows)}")
                time.sleep(0.01)

    vecs = np.vstack(all_vecs).astype("float32")
    d = vecs.shape[1]
    print("dim:", d)

    # índice: si normalizas, usa IP para cosine; si no, usa L2
    if args.normalize:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(vecs)
    faiss.write_index(index, str(index_path))
    print("OK índice:", index_path)
    print("OK meta:", meta_path)

if __name__ == "__main__":
    main()
