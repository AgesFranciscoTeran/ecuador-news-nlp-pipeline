#!/usr/bin/env bash
# instalar_ocr.sh — Instala el entorno del OCR (Docling + Surya) en el orden
# correcto. Es la única fuente de verdad del procedimiento: lo usan tanto el
# Dockerfile como la creación del venv.
#
#   ./instalar_ocr.sh                 # instala en el Python actual
#   ./instalar_ocr.sh /opt/venv       # instala en ese venv
#
# El orden importa: surya-ocr y docling declaran dependencias de torch y, si
# pip las resuelve, sustituyen la build de CUDA por una de CPU. El OCR seguiría
# funcionando, pero ~80 veces más lento y sin ningún error visible.
set -euo pipefail

VENV="${1:-}"
if [ -n "$VENV" ]; then
  PIP="$VENV/bin/pip"
  PY="$VENV/bin/python"
  [ -x "$PIP" ] || { echo "ERROR: no existe $PIP"; exit 1; }
else
  PIP="pip3"
  PY="python3"
fi

echo "── 1/4 PyTorch con CUDA 12.8 ──"
$PIP install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0 torchvision==0.26.0 \
  "cuda-bindings>=12.9.4,<13" "cuda-toolkit==12.8.1"

echo "── 2/4 Dependencias base ──"
$PIP install \
  pillow==10.4.0 \
  transformers==4.51.3 \
  "pydantic>=2.5.3" \
  "accelerate>=1.6.0"

echo "── 3/4 Surya + Docling (--no-deps para no tocar torch) ──"
$PIP install --no-deps surya-ocr==0.17.1
$PIP install --no-deps docling-surya==0.1.0
$PIP install --no-deps docling==2.103.0

echo "── 4/4 Dependencias de docling que el paso anterior omitió ──"
$PIP install \
  "docling-core>=2.19.0" \
  "docling-ibm-models>=3.3.5" \
  "docling-parse>=4.0.5" \
  "python-multipart>=0.0.18"

echo "── Verificación ──"
$PY - <<'PYEOF'
import torch, docling, docling_surya, transformers
print(f"  torch        {torch.__version__} | CUDA {torch.version.cuda} | "
      f"disponible: {torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}")
print(f"  docling      {docling.__version__}")
print("  docling_surya importado correctamente")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch no ve CUDA. Revisa el paso 1.")
PYEOF
$PIP check || echo "  (aviso: pip check reporta conflictos menores; el de typer es conocido)"
echo "Entorno OCR listo."