import os
import json
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling_surya import SuryaOcrOptions


# ---------------------------
# Config
# ---------------------------
IN_ROOT = Path(
    "/home/fteran/dhub/El Universo 2001-2022/El Universo 2001/01 enero-febrero/2001 01 enero 1-15"
)

OUT_ROOT = Path("/home/fteran/Francisco/docling test")


# ---------------------------
# Main
# ---------------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Buscar SOLO JPG originales
    images = sorted(IN_ROOT.glob("*.jpg")) + sorted(IN_ROOT.glob("*.JPG"))
    images = sorted(images)

    print(f"IN_ROOT: {IN_ROOT}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"Found {len(images)} images")

    if len(images) == 0:
        print("\n⚠️ No encontré JPG. Revisa el path:")
        print(IN_ROOT)
        return

    print("First 5:", [p.name for p in images[:5]])

    # Docling usando SuryaOCR plugin
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="suryaocr",
        allow_external_plugins=True,
        ocr_options=SuryaOcrOptions(lang=["es"]),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    for i, img in enumerate(images, 1):
        name = img.stem  # 00001
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(images)}] [IMG] {img.name}")

        try:
            # 1) Docling -> Markdown
            result = converter.convert(str(img))
            raw_md = result.document.export_to_markdown()
            (out_dir / "raw.md").write_text(raw_md, encoding="utf-8")

            # 2) Docling -> JSON layout
            try:
                doc_dict = result.document.export_to_dict()
                (out_dir / "docling.json").write_text(
                    json.dumps(doc_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                (out_dir / "docling_json_error.txt").write_text(
                    str(e), encoding="utf-8"
                )

        except Exception as e:
            (out_dir / "ERROR.txt").write_text(str(e), encoding="utf-8")
            print(f"ERROR con {img.name}: {e}")

    print("\nDONE Saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()
