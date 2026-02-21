# app.py
import json
import re
from pathlib import Path

import numpy as np
import requests
import faiss
import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Utils
# -----------------------------
WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{3,}")


def load_meta(meta_path: Path):
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ollama_embed_one(text: str, model: str, host: str) -> np.ndarray:
    url = host.rstrip("/") + "/api/embeddings"
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    v = np.array(r.json()["embedding"], dtype="float32")
    return v


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


def keyword_hits(text: str, qwords: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in qwords if w in t)


def read_text_if_exists(path_str: str, base_dir: str | None):
    if not base_dir:
        return None
    p = Path(base_dir) / path_str
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="ignore")
    return None


# --- Dedup helpers (near-duplicates) ---
def norm_for_dedup(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def shingles_words(text: str, k: int = 5) -> set:
    w = norm_for_dedup(text).split()
    if len(w) < k:
        return set()
    return set(tuple(w[i:i + k]) for i in range(len(w) - k + 1))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -----------------------------
# Caches
# -----------------------------
@st.cache_resource
def load_index_and_meta(index_dir: str):
    d = Path(index_dir)
    index = faiss.read_index(str(d / "faiss.index"))
    meta = load_meta(d / "meta.jsonl")
    return index, meta


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="U-Index Pipeline Explorer", layout="wide")
st.title("U-Index Pipeline Explorer (MVP)")

with st.sidebar:
    st.header("Conexión")
    idx_hq = st.text_input("Ruta índice HQ", value="Francisco/docling_test5_hq")
    idx_all = st.text_input("Ruta índice ALL", value="Francisco/docling_test5_all")
    use_hq = st.toggle("Usar HQ", value=True)
    index_dir = idx_hq if use_hq else idx_all

    st.divider()
    st.header("Embeddings")
    ollama_host = st.text_input("Ollama host", value="http://localhost:11434")
    emb_model = st.text_input("Modelo", value="nomic-embed-text")
    norm_query = st.toggle("Normalizar query (cosine)", value=True)

    st.divider()
    st.header("Archivos extra (opcional)")
    filter_report_path = st.text_input("Francisco/_demo_chunk_text/filter_report.json", value="Francisco/_demo_chunk_text/filter_report.json")
    docs_base_dir = st.text_input(
        "Base dir de .txt (para abrir doc completo)",
        value=""
    )  # ej: Francisco/_demo_text_final

tab_query, tab_analytics = st.tabs(["🔎 Query", "📊 Analytics"])

# -----------------------------
# Query tab
# -----------------------------
with tab_query:
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        q = st.text_area("Escribe tu consulta", value="crisis politica", height=80)
        topk = st.slider("Top K (FAISS)", 5, 80, 25)

        rerank = st.toggle("Rerank por hits de keywords (simple)", value=True)

        st.markdown("**Control de duplicados**")
        dedup_on = st.toggle("Dedup near-duplicates", value=True)
        dedup_thr = st.slider("Umbral dedup (Jaccard)", 0.60, 0.98, 0.85, 0.01)
        max_per_doc = st.slider("Máx resultados por documento", 1, 6, 2)

        if st.button("Buscar", type="primary"):
            try:
                index, meta = load_index_and_meta(index_dir)
            except Exception as e:
                st.error(f"No pude cargar índice/meta desde: {index_dir}\n\n{e}")
                st.stop()

            qwords = [w.lower() for w in WORD_RE.findall(q.lower())]
            v = ollama_embed_one(q, emb_model, ollama_host)
            if norm_query:
                v = normalize(v)

            D, I = index.search(v.reshape(1, -1), topk)
            D = D[0].tolist()
            I = I[0].tolist()

            rows = []
            for score, idx in zip(D, I):
                if idx < 0 or idx >= len(meta):
                    continue
                m = meta[idx]
                text = m.get("text", "")

                # Fallbacks: tu meta.jsonl puede usar otros nombres de campos
                doc_path = m.get("doc_path") or m.get("source_path") or m.get("path") or m.get("doc_id") or ""
                chunk_id = m.get("chunk_id") or m.get("chunk") or m.get("chunk_index") or m.get("id") or ""

                rows.append({
                    "score": float(score),
                    "doc_path": doc_path,
                    "chunk_id": chunk_id,
                    "quality_score": m.get("quality_score", None),
                    "hits": keyword_hits(text, qwords),
                    "snippet": (text[:320] + "…") if len(text) > 320 else text,
                    "_text_full": text,  # texto completo para dedup
                })

            df = pd.DataFrame(rows)

            if df.empty:
                st.warning("No hubo resultados (df vacío). Revisa índice/meta.")
                st.stop()

            # Rerank simple: combina score FAISS con hits
            if rerank:
                df["rerank"] = df["score"] + 0.08 * df["hits"]
                df = df.sort_values(["rerank", "score"], ascending=False)
            else:
                df = df.sort_values("score", ascending=False)

            # Dedup + límite por doc
            if dedup_on:
                kept = []
                kept_shingles = []
                per_doc = {}

                for _, r in df.iterrows():
                    doc = r.get("doc_path", "") or ""
                    per_doc[doc] = per_doc.get(doc, 0)

                    # límite de resultados por doc
                    if per_doc[doc] >= max_per_doc:
                        continue

                    txt = r.get("_text_full", "") or r.get("snippet", "")
                    sh = shingles_words(txt, k=5)

                    # filtrar near-duplicates
                    too_similar = any(jaccard(sh, s2) >= dedup_thr for s2 in kept_shingles)
                    if too_similar:
                        continue

                    kept.append(r)
                    kept_shingles.append(sh)
                    per_doc[doc] += 1

                df = pd.DataFrame(kept)

            # Limpia columnas internas (no mostrar _text_full)
            show_df = df.drop(columns=["_text_full"], errors="ignore")

            st.subheader("Resultados")
            st.dataframe(show_df, use_container_width=True, height=420)

            st.subheader("Vista rápida")
            if not df.empty:
                pick = st.selectbox(
                    "Selecciona un resultado",
                    df.index.tolist(),
                    format_func=lambda i: f"{df.loc[i,'doc_path']} :: {df.loc[i,'chunk_id']}"
                )

                st.markdown("**Snippet**")
                st.code(df.loc[pick, "snippet"])

                full = read_text_if_exists(df.loc[pick, "doc_path"], docs_base_dir.strip() or None)
                if full:
                    st.markdown("**Documento completo (preview)**")
                    st.text_area("doc", full[:6000], height=260)
                else:
                    st.caption("Tip: llena 'Base dir de .txt' en el sidebar para poder abrir el documento completo.")

    with colB:
        st.subheader("Checklist visual (rápido)")
        st.write("✔ ¿Los top resultados son coherentes?")
        st.write("✔ ¿Ves chunks duplicados por overlap? (dedup debería reducirlo)")
        st.write("✔ ¿Hay texto pegado tipo `solupersonas` (layout/columnas)?")
        st.write("✔ ¿HQ realmente mejora vs ALL?")
        st.caption("Este MVP es para *ver* problemas antes de optimizar.")

# -----------------------------
# Analytics tab
# -----------------------------
with tab_analytics:
    st.subheader("Calidad y filtros")
    fr = None
    try:
        p = Path(filter_report_path)
        if p.exists():
            fr = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        fr = None

    if not fr:
        st.info("No encontré/parseé filter_report.json. Ajusta la ruta en el sidebar si lo quieres graficar.")
    else:
        # Razones de drop
        reasons = fr.get("drop_reasons") or fr.get("reasons") or {}
        if reasons:
            rdf = pd.DataFrame(
                {"reason": list(reasons.keys()), "count": list(reasons.values())}
            ).sort_values("count", ascending=False)
            fig = px.bar(
                rdf.head(12),
                x="reason",
                y="count",
                title="Top razones de descarte (drop_reasons)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Resumen general
        kept = fr.get("kept", None)
        dropped = fr.get("dropped", None)
        thr = fr.get("threshold", None)
        st.markdown(f"- **threshold:** `{thr}`  \n- **kept:** `{kept}`  \n- **dropped:** `{dropped}`")

        # Score summary
        ss = fr.get("score_summary", {})
        if ss:
            st.markdown(
                f"- **quality_score (kept)** → min `{ss.get('min')}`, avg `{ss.get('avg')}`, max `{ss.get('max')}`"
            )
