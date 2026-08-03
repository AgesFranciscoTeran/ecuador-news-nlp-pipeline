#!/usr/bin/env bash
set -uo pipefail
cd ~/Proyecto_DataHub || exit 1
PERIODICOS=("El Comercio 2001-2022" "El Telégrafo 2001-2019" "El Universo 2001-2022")

echo "== 1) integridad de cada archivo =="
fallos=0; n=0
for f in imagenes_periodico_procesadas/ocr_output/*.tar.gz; do
  n=$((n+1))
  gzip -t "$f" 2>/dev/null || { echo "   CORRUPTO: $f"; fallos=$((fallos+1)); }
done
echo "   $n archivos | $fallos corruptos"
ls imagenes_periodico_procesadas/ocr_output/*.parcial 2>/dev/null && echo "   OJO: quedaron .parcial sin terminar"

echo "== 2) ningún año quedó fuera =="
faltan=0
for p in "${PERIODICOS[@]}"; do
  [ -d "$p" ] || continue
  for anio in "$p"/*/; do
    base=$(basename "$anio")
    t="imagenes_periodico_procesadas/ocr_output/$(echo "$base" | tr ' ' '_').tar.gz"
    [ -f "$t" ] || { echo "   FALTA: $base"; faltan=$((faltan+1)); }
  done
done
echo "   años sin empaquetar: $faltan"

echo "== 3) conteo de archivos: origen vs empaquetado =="
orig=$(find "${PERIODICOS[@]}" -type f 2>/dev/null | wc -l)
emp=$(for f in imagenes_periodico_procesadas/ocr_output/*.tar.gz; do tar -tzf "$f"; done | grep -cv '/$')
echo "   en disco: $orig | dentro de los tar: $emp | diferencia: $((orig-emp))"

echo "== 4) checksums para validar tras la transferencia =="
( cd imagenes_periodico_procesadas/ocr_output && sha256sum *.tar.gz > SHA256SUMS.txt )
echo "   $(wc -l < imagenes_periodico_procesadas/ocr_output/SHA256SUMS.txt) hashes en imagenes_periodico_procesadas/ocr_output/SHA256SUMS.txt"
[ "$fallos" -eq 0 ] && [ "$faltan" -eq 0 ] && [ "$orig" -eq "$emp" ] \
  && echo "RESULTADO: todo correcto" || echo "RESULTADO: revisar lo marcado arriba"
