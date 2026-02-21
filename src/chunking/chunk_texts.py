from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional


# ----------------------------
# Tokenizer (tiktoken optional)
# ----------------------------
def try_get_tiktoken(encoding_name: str):
    try:
        import tiktoken  # type: ignore
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None


def count_tokens(text: str, enc) -> int:
    if enc is None:
        # fallback: approx tokens by words
        return len(text.split())
    return len(enc.encode(text))


def slice_with_overlap(tokens: List[int], start: int, end: int, overlap: int) -> Tuple[int, int]:
    # next chunk starts "overlap" tokens before end
    next_start = max(end - overlap, start)
    return next_start, end


# ----------------------------
# Chunking logic
# ----------------------------
PARA_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.MULTILINE)

def split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paras = [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]
    return paras if paras else [text]


def chunk_by_paragraphs(
    paragraphs: List[str],
    max_tokens: int,
    overlap_tokens: int,
    enc
) -> List[str]:
    """
    Acumula párrafos hasta max_tokens. Si un párrafo es demasiado grande,
    se parte internamente por tokens.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for p in paragraphs:
        p_tokens = count_tokens(p, enc)

        # Si el párrafo solo ya excede el límite, cortarlo por tokens
        if p_tokens > max_tokens:
            # flush lo acumulado antes
            if current:
                chunks.append("\n\n".join(current).strip())
                current, current_tokens = [], 0

            if enc is None:
                # fallback por palabras
                words = p.split()
                i = 0
                while i < len(words):
                    chunk_words = words[i:i + max_tokens]
                    chunks.append(" ".join(chunk_words).strip())
                    i = max(i + max_tokens - overlap_tokens, i + 1)
            else:
                ids = enc.encode(p)
                i = 0
                while i < len(ids):
                    j = min(i + max_tokens, len(ids))
                    chunks.append(enc.decode(ids[i:j]).strip())
                    i = max(j - overlap_tokens, i + 1)

            continue

        # Si agregar este párrafo excede, flush y empezar nuevo
        if current_tokens + p_tokens > max_tokens and current:
            chunks.append("\n\n".join(current).strip())
            current, current_tokens = [], 0

        current.append(p)
        current_tokens += p_tokens

    if current:
        chunks.append("\n\n".join(current).strip())

    # Overlap entre chunks (solo si hay tiktoken). En fallback por palabras,
    # ya hay overlap dentro de párrafos grandes; aquí lo dejamos simple.
    if enc is not None and overlap_tokens > 0 and len(chunks) > 1:
        # reconstruye con overlap token-level para coherencia
        rebuilt: List[str] = []
        prev_tail_ids: List[int] = []

        for idx, ch in enumerate(chunks):
            ids = enc.encode(ch)
            if idx == 0:
                rebuilt.append(ch)
            else:
                # anteponer tail del chunk anterior
                prefix = enc.decode(prev_tail_ids).strip()
                if prefix:
                    rebuilt.append((prefix + "\n" + ch).strip())
                else:
                    rebuilt.append(ch)
            prev_tail_ids = ids[-overlap_tokens:] if len(ids) > overlap_tokens else ids

        chunks = rebuilt

    return [c for c in chunks if c.strip()]


# ----------------------------
# IO
# ----------------------------
def write_chunks_txt(out_dir: Path, rel_path: Path, chunks: List[str]) -> None:
    base = rel_path.with_suffix("")  # remove .txt
    target_dir = out_dir / base
    target_dir.mkdir(parents=True, exist_ok=True)

    for i, ch in enumerate(chunks):
        (target_dir / f"chunk_{i:04d}.txt").write_text(ch + "\n", encoding="utf-8")


def append_chunks_jsonl(jsonl_path: Path, rel_path: Path, chunks: List[str]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            rec = {
                "source_path": str(rel_path),
                "chunk_index": i,
                "text": ch
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directorio con txt limpios (ej: Francisco/_demo_text_final)")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida (ej: Francisco/docling_test2)")
    ap.add_argument("--format", choices=["txt", "jsonl"], default="jsonl",
                    help="Salida: txt (1 archivo por chunk) o jsonl (recomendado)")
    ap.add_argument("--max_tokens", type=int, default=500, help="Tamaño del chunk (tokens si tiktoken, si no palabras)")
    ap.add_argument("--overlap", type=int, default=80, help="Overlap entre chunks")
    ap.add_argument("--encoding", default="cl100k_base",
                    help="Encoding tiktoken (cl100k_base suele funcionar bien)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = try_get_tiktoken(args.encoding)

    jsonl_path: Optional[Path] = None
    if args.format == "jsonl":
        jsonl_path = out_dir / "chunks.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()  # rehacer limpio

    txt_files = list(in_dir.rglob("*.md"))
    print(f"Encontrados {len(txt_files)} txts en {in_dir}")

    for p in txt_files:
        rel = p.relative_to(in_dir)
        raw = p.read_text(encoding="utf-8", errors="replace")

        paras = split_paragraphs(raw)
        chunks = chunk_by_paragraphs(paras, args.max_tokens, args.overlap, enc)

        if args.format == "txt":
            write_chunks_txt(out_dir, rel, chunks)
        else:
            append_chunks_jsonl(jsonl_path, rel, chunks)

    print("OK. Chunks guardados en:", out_dir)


if __name__ == "__main__":
    main()
