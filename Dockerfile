# Dockerfile — entorno del OCR (Docling + Surya).
#
# Reemplaza la práctica anterior de `docker commit` sobre un contenedor
# configurado a mano: con esto la imagen se reconstruye igual en cualquier
# máquina y el repositorio deja de depender de un binario irreproducible.
#
#   docker build -t datahub-ocr:latest -f Dockerfile .
#
# La instalación NO se hace aquí: se delega en instalar_ocr.sh, que respeta el
# orden y los --no-deps necesarios. Así el venv del host y la imagen se
# construyen exactamente igual.
#
# Los scripts (runner/supervisor.sh, docling_suryaOCR.py) no se copian: se
# montan en tiempo de ejecución desde el host para poder editarlos sin
# reconstruir la imagen. Ver runner/datahub_ctl.sh.

# Base ubuntu24.04 (GLIBC 2.39, Python 3.12) y no 22.04 (GLIBC 2.35, Python
# 3.10): la guía del entorno exige GLIBC >= 2.38 y Python 3.12 para los wheels
# modernos. CUDA 12.8 es coherente con cuda-toolkit==12.8.1.
# PENDIENTE: verificar con un build real antes de confiar en esta imagen.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# libgl1 y libglib2.0-0 los necesita OpenCV, que arrastra docling.
# En 24.04 el paquete es libgl1 (en 22.04 se llamaba libgl1-mesa-glx).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git ca-certificates \
        libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

COPY instalar_ocr.sh /tmp/instalar_ocr.sh
RUN chmod +x /tmp/instalar_ocr.sh && /tmp/instalar_ocr.sh && rm /tmp/instalar_ocr.sh

WORKDIR /data/out