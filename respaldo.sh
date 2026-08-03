#!/usr/bin/env bash
# respaldo.sh — Empaqueta el entregable (bases empalmadas + base vectorial)
# hacia el disco duro, con checksums y un manifiesto que explica qué es cada
# cosa. Pensado para que alguien que recibe el disco entienda el contenido sin
# preguntarte nada.
#
#   ./respaldo.sh /media/fteran/DISCO/DataHub_entrega          # copia tal cual
#   ./respaldo.sh /media/fteran/DISCO/DataHub_entrega --comprimir   # CSV en .gz
#
# Idempotente: se puede volver a correr; rsync solo transfiere lo que cambió.
set -euo pipefail

DEST="${1:?uso: ./respaldo.sh /ruta/al/disco/DataHub_entrega [--comprimir]}"
COMPRIMIR="${2:-}"
SRC="$HOME/Proyecto_DataHub"
FECHA="$(date +%Y-%m-%d)"

mkdir -p "$DEST"/{bases,vectordb,scripts,informes}

copiar() {  # copiar <origen> <destino>   (barra final = copiar contenido)
  local src="$1" dst="$2"
  [ -e "${src%/}" ] || { echo "  [skip] no existe: $src"; return 0; }
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$src" "$dst"
  elif [ "$src" != "${src%/}" ]; then
    cp -a "${src%/}/." "$dst"
  else
    cp -a "$src" "$dst"
  fi
}

echo "== 1/4 bases de datos =="
if [ "$COMPRIMIR" = "--comprimir" ]; then
  for f in "$SRC"/csv_corpus/corpus_maestro.csv \
           "$SRC"/csv_corpus/corpus_limpio.csv \
           "$SRC"/csv_webscraping/articulos_*.csv; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    if [ ! -f "$DEST/bases/$base.gz" ]; then
      echo "  comprimiendo $base"
      gzip -c "$f" > "$DEST/bases/$base.gz"
    fi
  done
else
  copiar "$SRC/csv_corpus/corpus_maestro.csv" "$DEST/bases/"
  copiar "$SRC/csv_corpus/corpus_limpio.csv"  "$DEST/bases/"
  for f in "$SRC"/csv_webscraping/articulos_*.csv; do
    copiar "$f" "$DEST/bases/"
  done
fi

echo "== 2/4 base vectorial =="
copiar "$SRC/vectordb/" "$DEST/vectordb/"

echo "== 3/4 scripts e informes (reproducibilidad) =="
for s in corpus_to_csv.py limpiar_corpus.py auditar_calidad.py \
         cobertura_temporal.py empalmar_csv.py senales_diarias.py \
         construir_vectordb.py consultar_vectordb.py respaldo.sh; do
  copiar "$SRC/runner/$s" "$DEST/scripts/"
done
for d in "$SRC"/cobertura* "$SRC"/informe_calidad* "$SRC"/senales; do
  [ -d "$d" ] && copiar "$d" "$DEST/informes/"
done

echo "== 4/4 manifiesto y checksums =="
{
  echo "# Entrega DataHub — corpus de prensa ecuatoriana"
  echo
  echo "Generado: $FECHA"
  echo
  echo "## Contenido"
  echo
  echo "- \`bases/corpus_maestro.csv\` — corpus unificado (webscraping + OCR),"
  echo "  deduplicado. Columnas: id, origen, id_original, periodico, fecha,"
  echo "  fecha_fuente, seccion, titulo, texto, flags, path."
  echo "- \`bases/corpus_limpio.csv\` — solo OCR, ya limpio (insumo del maestro)."
  echo "- \`bases/articulos_*.csv\` — webscraping original, sin tocar."
  echo "- \`vectordb/\` — base vectorial LanceDB (directorio de archivos: se abre"
  echo "  con \`lancedb.connect(<ruta>)\`, no necesita servidor). Incluye"
  echo "  \`manifiesto.json\` con los meses ya indexados."
  echo "- \`scripts/\` — pipeline completo, en orden de ejecución."
  echo "- \`informes/\` — cobertura temporal, auditoría de calidad y señales."
  echo
  echo "## Trazabilidad de fechas (columna fecha_fuente)"
  echo
  echo "| etiqueta | significado | confianza |"
  echo "|---|---|---|"
  echo "| web | fecha del webscraping | día exacto |"
  echo "| header | fecha leída del encabezado de la página | día exacto |"
  echo "| pagina_anterior / retro | heredada de la página ancla del mismo día | día exacto |"
  echo "| cuerpo | fecha rescatada del cuerpo, validada con el día de la semana | día exacto |"
  echo "| cuerpo_ant / cuerpo_retro | heredada de un ancla de cuerpo | día probable |"
  echo "| bloque_mes / bloque_quincena | solo se conoce el mes/quincena | agregados mensuales |"
  echo
  echo "## Tamaños y conteos"
  echo '```'
  for f in "$DEST"/bases/*; do
    [ -e "$f" ] || continue
    n="?"
    case "$f" in
      *.csv) n="$(( $(wc -l < "$f") - 1 ))" ;;
      *.csv.gz) n="$(( $(gzip -cd "$f" | wc -l) - 1 ))" ;;
    esac
    printf '%-42s %10s  %s filas\n' "$(basename "$f")" \
      "$(du -h "$f" | cut -f1)" "$n"
  done
  echo '```'
  echo
  echo "## Verificar integridad"
  echo '```'
  echo "cd $(basename "$DEST") && sha256sum -c SHA256SUMS.txt"
  echo '```'
} > "$DEST/MANIFIESTO.md"

cd "$DEST"
find . -type f ! -name SHA256SUMS.txt -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
echo
echo "[ok] entrega en $DEST"
echo "     $(wc -l < SHA256SUMS.txt) archivos | $(du -sh . | cut -f1) en disco"
echo "     revisa MANIFIESTO.md"
