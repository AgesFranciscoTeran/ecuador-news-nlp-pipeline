from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

LETTER_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
DIGIT_RE  = re.compile(r"\d")
SPACE_RE  = re.compile(r"\s")

# patrones comunes de “ruido estructural
MANY_SEPARATORS_RE = re.compile(r"[-_=]{5,}|[|]{3,}")
PHONEISH_RE = re.compile(r"\b(Tel|Telf|PBX|Fax)\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"%")
MONEY_RE = re.compile(r"[$€]|USD|US\$", re.IGNORECASE)

def stats(text: str) -> dict:
    t = text.strip()
    n = len(t)
    if n == 0:
        return {
            "n": 0, "alpha": 0, "digit": 0, "space": 0,
            "alpha_ratio": 0.0, "digit_ratio": 0.0, "space_ratio": 0.0,
            "lines": 0, "avg_line_len": 0.0, "short_lines_ratio": 1.0,
        }

    alpha = len(LETTER_RE.findall(t))
    digit = len(DIGIT_RE.findall(t))
    space = len(SPACE_RE.findall(t))

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    num_lines = len(lines)
    avg_line_len = sum(len(ln) for ln in lines) / max(num_lines, 1)
    short_lines = sum(1 for ln in lines if len(ln) <= 20)
    short_lines_ratio = short_lines / max(num_lines, 1)

    return {
        "n": n,
        "alpha": alpha,
        "digit": digit,
        "space": space,
        "alpha_ratio": alpha / n,
        "digit_ratio": digit / n,
        "space_ratio": space / n,
        "lines": num_lines,
        "avg_line_len": avg_line_len,
        "short_lines_ratio": short_lines_ratio,
    }

def quality_score(s: dict, text: str) -> tuple[float, list[str]]:
    """
    Score 0..1. Heurística para embeddings:
    - premia texto con letras (contenido semántico)
    - penaliza muchos dígitos/separadores/listados
    - penaliza chunks con muchas líneas cortas (tablas/listas)
    """
    reasons = []

    if s["n"] == 0:
        return 0.0, ["empty"]

    score = 0.0

    # base por señal semántica
    score += min(0.85, 1.6 * s["alpha_ratio"])

    # penalizaciones
    if s["digit_ratio"] > 0.18:
        score -= (s["digit_ratio"] - 0.18) * 1.8
        reasons.append("digit_heavy")

    if s["short_lines_ratio"] > 0.55 and s["lines"] >= 6:
        score -= (s["short_lines_ratio"] - 0.55) * 1.2
        reasons.append("many_short_lines")

    if MANY_SEPARATORS_RE.search(text):
        score -= 0.15
        reasons.append("many_separators")

    # estas señales son típicas de anuncios/listados; no siempre, pero ayudan
    if PHONEISH_RE.search(text):
        score -= 0.10
        reasons.append("phoneish")

    if PERCENT_RE.search(text) and s["digit_ratio"] > 0.12:
        score -= 0.08
        reasons.append("percent+digits")

    if MONEY_RE.search(text) and s["digit_ratio"] > 0.10:
        score -= 0.06
        reasons.append("money+digits")

    # clamp
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0

    return score, reasons

def sha1_norm(text: str) -> str:
    # normaliza espacios para dedup exacta “robusta”
    t = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()

def main():
    in_path = Path("Francisco/docling_test3/chunks.jsonl")
    out_path = Path("Francisco/docling_test4/chunks_filtered.jsonl")
    report_path = Path("Francisco/docling_test4/filter_report.json")

    # umbral por defecto (ajustable): 0.55 suele botar tablas/listados sin matar prosa
    THRESH = 0.55

    seen_hashes = set()
    kept = 0
    dropped = 0
    drop_reasons = Counter()
    scores = []
    per_doc_kept = defaultdict(int)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                dropped += 1
                drop_reasons["empty_line"] += 1
                continue

            obj = json.loads(line)
            text = obj.get("text", "")
            s = stats(text)
            score, reasons = quality_score(s, text)

            h = sha1_norm(text)
            is_dup = h in seen_hashes
            if not is_dup:
                seen_hashes.add(h)

            keep = (score >= THRESH) and (not is_dup) and (s["n"] >= 80)  # 80 chars mínimo

            if keep:
                kept += 1
                scores.append(score)
                per_doc_kept[obj["source_path"]] += 1

                out_obj = {
                    "source_path": obj["source_path"],
                    "chunk_index": obj["chunk_index"],
                    "quality_score": round(score, 4),
                    "text": text
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            else:
                dropped += 1
                if is_dup:
                    drop_reasons["duplicate"] += 1
                if s["n"] < 80:
                    drop_reasons["too_short"] += 1
                for r in reasons:
                    drop_reasons[r] += 1
                if score < THRESH:
                    drop_reasons["below_threshold"] += 1

    # reporte
    report = {
        "input": str(in_path),
        "output": str(out_path),
        "threshold": THRESH,
        "kept": kept,
        "dropped": dropped,
        "kept_unique_docs": len(per_doc_kept),
        "docs_with_zero_kept_chunks": None,  # lo calculamos luego si quieres
        "drop_reasons_top": drop_reasons.most_common(15),
        "score_summary": {
            "count": len(scores),
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "avg": (sum(scores) / len(scores)) if scores else None,
        }
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK")
    print("Kept:", kept, "| Dropped:", dropped)
    print("Output:", out_path)
    print("Report:", report_path)

if __name__ == "__main__":
    main()
