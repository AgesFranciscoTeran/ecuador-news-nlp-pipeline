from pathlib import Path
import json, re, gzip

LETTERS = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]")
ONLY_NUMBERS = re.compile(r"^\s*[\d\.,:%\-\+\(\) ]+\s*$")

def clean_ocr_text(text: str, min_len_letters_line: int = 10) -> str:
    # une pala-\nbra -> palabra
    text = re.sub(r"([A-Za-zÁÉÍÓÚáéíóúÑñÜü])-\n([A-Za-zÁÉÍÓÚáéíóúÑñÜü])", r"\1\2", text)

    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue

        has_letters = bool(LETTERS.search(s))
        is_numbers = bool(ONLY_NUMBERS.match(s))

        # borra líneas cortas SOLO si tienen letras
        if has_letters and len(s) < min_len_letters_line:
            continue

        # si no tiene letras ni es números, probablemente basura
        if (not has_letters) and (not is_numbers):
            cleaned = re.sub(r"[^\d\w\s\.,:%\-\+\(\)]", "", s).strip()
            if len(cleaned) < 4:
                continue
            s = cleaned

        out.append(s)

    return "\n".join(out)

def line_key(tl: dict):
    # orden aproximado: top-to-bottom, left-to-right usando polygon
    poly = tl.get("polygon") or []
    if not poly:
        return (10**9, 10**9)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(ys), min(xs))

def extract_text_from_item(item: dict, conf_min: float = 0.0) -> str:
    lines = item.get("text_lines") or []
    # filtra por confianza si quieres
    kept = []
    for tl in lines:
        if not isinstance(tl, dict):
            continue
        if tl.get("confidence", 1.0) < conf_min:
            continue
        txt = (tl.get("text") or "").strip()
        if txt:
            kept.append(tl)

    kept.sort(key=line_key)
    return "\n".join((tl["text"].strip() for tl in kept if tl.get("text")))

def process_results_json(results_json: Path, out_dir: Path, conf_min: float = 0.0, compress=True):
    data = json.loads(results_json.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_id, lst in data.items():
        if not isinstance(lst, list) or not lst:
            continue
        item = lst[0]
        if not isinstance(item, dict):
            continue

        text = extract_text_from_item(item, conf_min=conf_min)
        text = clean_ocr_text(text)

        if text.strip():
            (out_dir / f"{img_id}.txt").write_text(text, encoding="utf-8")
            n += 1

    print(f"OK: {results_json} -> {n} txt en {out_dir}")

    if compress:
        gz = results_json.with_suffix(".json.gz")
        if not gz.exists():
            with gzip.open(gz, "wb") as f:
                f.write(results_json.read_bytes())
        # Si quieres ahorrar espacio, descomenta:
        # results_json.unlink()

if __name__ == "__main__":
    OCR_FOLDER = Path("Francisco/_demo_surya_out/El Universo 2001/01 enero-febrero/2001 01 enero 1-15/2001 01 enero 1-15")
    results_json = OCR_FOLDER / "results.json"

    OUT_TXT = Path("Francisco/_demo_text_clean_split/El Universo 2001/01 enero-febrero/2001 01 enero 1-15/2001 01 enero 1-15")
    process_results_json(results_json, OUT_TXT, conf_min=0.0, compress=True)
