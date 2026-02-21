import json
from pathlib import Path
import numpy as np
import requests
import faiss
import re
import unicodedata

# -----------------------------
# Text normalization
# -----------------------------
def norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

# -----------------------------
# Embedding (Ollama)
# -----------------------------
def embed_one(q, model="nomic-embed-text", host="http://localhost:11434"):
    r = requests.post(
        host.rstrip("/") + "/api/embeddings",
        json={"model": model, "prompt": q},
        timeout=120,
    )
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v

def load_meta(meta_path: Path):
    return [json.loads(x) for x in meta_path.read_text(encoding="utf-8").splitlines() if x.strip()]

# -----------------------------
# Rerank helpers
# -----------------------------
STOP = {norm(x) for x in {
    "ecuador","quito","guayaquil",
    "enero","febrero","marzo","abril","mayo","junio","julio","agosto",
    "septiembre","octubre","noviembre","diciembre",
    "lunes","martes","miércoles","jueves","viernes","sábado","domingo",
    "año","años","del","de","la","el","los","las","para","por","con","una","un","y","o","en","al","se","su","sus",
    "1999","2000","2001","2002","2003","2004","2005",
}}

OFFICIAL_HINT = re.compile(
    r"\b(ing\.|ministro|ministra|ministerio|obras publicas|comunicaciones|republica|presidente constitucional)\b",
    re.IGNORECASE,
)

def informative_qwords(query: str):
    qn = norm(query)
    qwords = [w for w in re.findall(r"\w+", qn) if len(w) >= 4]
    qwords = [w for w in qwords if w not in STOP]
    seen = set()
    out = []
    for w in qwords:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out

def keyword_hits_norm(text_norm: str, qwords) -> int:
    if not qwords:
        return 0
    return sum(1 for w in qwords if w in text_norm)

def rerank_score(base: float, text: str, qwords) -> tuple[float, int]:
    tn = norm(text)
    hits = keyword_hits_norm(tn, qwords)

    score = base + 0.03 * hits

    # Si el query incluye "politica", exigir señal política ("politic" cubre politica/politico/politicos)
    if "politica" in qwords and "politic" not in tn:
        score -= 0.35

    # Si hay 2+ keywords informativas, penaliza fuerte si solo pega 0-1
    if len(qwords) >= 2 and hits < 2:
        score -= 0.20

    # Sin hits -> baja extra
    if qwords and hits == 0:
        score -= 0.10

    # Oficios/plantillas -> baja suave
    if OFFICIAL_HINT.search(tn):
        score -= 0.05

    return score, hits

# -----------------------------
# Retrieval: top docs + top chunks (HITS-AWARE DOC SCORE)
# -----------------------------
def search_docs_top_chunks(index_dir, query, k_docs=5, chunks_per_doc=2, oversample=400):
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta  = load_meta(index_dir / "meta.jsonl")

    qv = embed_one(query)
    D, I = index.search(qv, oversample)

    qwords = informative_qwords(query)
    need = len(qwords)

    per_doc = {}
    for j in range(oversample):
        idx = int(I[0][j])
        base = float(D[0][j])
        m = meta[idx]
        doc = m["source_path"]

        score, hits = rerank_score(base, m["text"], qwords)
        per_doc.setdefault(doc, [])
        per_doc[doc].append((score, hits, m))

    doc_items = []
    for doc, lst in per_doc.items():
        # ordenar por score
        lst.sort(key=lambda x: x[0], reverse=True)

        # prioridad por cobertura de qwords
        if need > 0:
            full = [x for x in lst if x[1] >= need]          # hits >= need (ej. 2/2)
            partial = [x for x in lst if x[1] == need - 1]   # hits = need-1 (ej. 1/2)
        else:
            full, partial = [], []

        if full:
            doc_score = full[0][0]
            top_chunks = full[:chunks_per_doc]
        elif partial:
            doc_score = partial[0][0]
            top_chunks = partial[:chunks_per_doc]
        else:
            doc_score = lst[0][0]
            top_chunks = lst[:chunks_per_doc]

        doc_items.append((doc_score, doc, top_chunks, qwords))

    doc_items.sort(key=lambda x: x[0], reverse=True)
    return doc_items[:k_docs]

def query_with_fallback(query: str,
                        hq_dir="Francisco/docling_test5_hq",
                        all_dir="Francisco/docling_test5_all",
                        k_docs=5, chunks_per_doc=2):
    res_hq = search_docs_top_chunks(hq_dir, query, k_docs=k_docs, chunks_per_doc=chunks_per_doc, oversample=400)
    
    # criterio simple: si el top HQ es muy bajo, intenta ALL
    if (not res_hq) or (res_hq[0][0] < 0.45):
        res_all = search_docs_top_chunks(all_dir, query, k_docs=k_docs, chunks_per_doc=chunks_per_doc, oversample=700)
        return "ALL", res_all
    return "HQ", res_hq

# ---- run ----
q = "crisis política en ecuador 2001"
which, res = query_with_fallback(q, k_docs=5, chunks_per_doc=2)

print(f"{which} (docs, top chunks) — reranked:")
for doc_score, doc, chunks, qwords in res:
    print("\n===", doc, "doc_score:", doc_score, "| qwords:", qwords)
    for s, hits, m in chunks:
        preview = m["text"][:500].replace("\n", " ")
        print(" ", f"score={s:.4f}", f"hits={hits}", m["id"])
        print("    ", preview, "...")
