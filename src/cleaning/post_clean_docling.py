from pathlib import Path
import re
import html

DOC_IN_ROOT = Path("/home/fteran/Francisco/docling test")
DOC_OUT_ROOT = Path("/home/fteran/Francisco/docling_test2")
DOC_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# quitar comentarios docling
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# convertir &lt;math&gt; a <math> y eliminar tags <math>
MATH_TAG_RE = re.compile(r"</?math>", re.IGNORECASE)

# (opcional) otros tags que podrían colarse
GENERIC_TAG_RE = re.compile(r"</?(sup|sub|span|b|i|u|em|strong)>", re.IGNORECASE)

# remover web y dirección en internet
WEB_LINE_RE = re.compile(
    r"^\s*(DIRECCIÓN EN INTERNET:|DIRECCION EN INTERNET:)?\s*www\.eluniverso\.com\s*$",
    re.IGNORECASE,
)

# detectar bloque índice (lo removemos completo)
IDX_START_RE = re.compile(r"^\s*##\s*INDICE\s*$", re.IGNORECASE)

# stop cuando termina el índice (cuando vuelve a otra sección fuerte)
IDX_END_RE = re.compile(
    r"^\s*##\s*(Ayuda a|EDC|Se incrementó|Bancomex|Mas provisiones|Recaudaciones|out clausuro)\b",
    re.IGNORECASE,
)

def soft_clean_raw(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    # 0) decode HTML entities (&lt; &gt; &amp; etc.)
    md = html.unescape(md)

    # 0.1) elimina tags <math>...</math> manteniendo contenido
    md = MATH_TAG_RE.sub("", md)

    # (opcional) limpia algunos tags simples si aparecen
    md = GENERIC_TAG_RE.sub("", md)

    # 1) quitar <!-- image -->
    md = HTML_COMMENT_RE.sub("", md)

    lines = md.splitlines()
    out = []

    skipping_index = False

    for ln in lines:
        t = ln.rstrip()

        # 2) quitar línea web
        if WEB_LINE_RE.match(t.strip()):
            continue

        # 3) remover bloque índice completo
        if IDX_START_RE.match(t.strip()):
            skipping_index = True
            continue

        if skipping_index:
            # termina cuando aparece una sección grande de nuevo
            if IDX_END_RE.match(t.strip()):
                skipping_index = False
                out.append(t)  # incluir la línea que reanuda contenido
            continue

        out.append(t)

    # limpiar saltos múltiples excesivos
    text = "\n".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip() + "\n"

def main():
    subdirs = sorted([p for p in DOC_IN_ROOT.iterdir() if p.is_dir()])

    total = 0
    written = 0

    for d in subdirs:
        raw_md_path = d / "raw.md"
        if not raw_md_path.exists():
            continue

        total += 1
        raw = raw_md_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = soft_clean_raw(raw)

        out_file = DOC_OUT_ROOT / f"{d.name}_rawclean.md"
        out_file.write_text(cleaned, encoding="utf-8")
        written += 1

    print("OK. Procesados:", written, "/", total)
    print("OUT:", DOC_OUT_ROOT)

if __name__ == "__main__":
    main()
