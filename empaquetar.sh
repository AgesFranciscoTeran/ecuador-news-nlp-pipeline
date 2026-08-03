#!/usr/bin/env bash
set -uo pipefail
cd ~/Proyecto_DataHub || exit 1
mkdir -p imagenes_periodico_procesadas/ocr_output
COMPRESOR=$(command -v pigz || command -v gzip)
for periodico in "El Comercio 2001-2022" "El Telégrafo 2001-2019" "El Universo 2001-2022"; do
  [ -d "$periodico" ] || continue
  for anio in "$periodico"/*/; do
    base=$(basename "$anio")
    salida="imagenes_periodico_procesadas/ocr_output/$(echo "$base" | tr ' ' '_').tar.gz"
    [ -f "$salida" ] && { echo "ya existe: $salida"; continue; }
    echo "empaquetando $base ..."
    if tar -cf - -C "$periodico" "$base" | "$COMPRESOR" -6 > "$salida.parcial"; then
      mv "$salida.parcial" "$salida"
      echo "  ok: $(du -h "$salida" | cut -f1)"
    else
      echo "  FALLÓ $base"; rm -f "$salida.parcial"
    fi
  done
done
echo "LISTO"; du -sh imagenes_periodico_procesadas/ocr_output
