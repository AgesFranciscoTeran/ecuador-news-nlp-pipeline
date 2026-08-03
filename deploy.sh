#!/usr/bin/env bash
# deploy.sh — Despliegue completo de datahub-ocr
# Ejecutar como root (o con sudo) desde /home/fteran/Proyecto_DataHub
#
# Uso:  ./deploy.sh              # modo interactivo (pregunta antes de cada paso)
#       ./deploy.sh --yes        # modo automático (sin confirmar)
set -u

BASE=/home/fteran/Proyecto_DataHub
COLAB=colab_fteran
IMAGE=datahub-ocr:latest
ME=$(basename "$0")

AUTO=0
[ "${1:-}" = "--yes" ] && AUTO=1

confirm() {
  local msg="$1"
  if [ "$AUTO" -eq 1 ]; then
    echo "  → $msg (automático)"
    return 0
  fi
  echo ""
  read -r -p "  ⚠ $msg  [Enter/S] para saltar, [y] para sí, [n] para no: " resp
  case "$resp" in
    [yY]) return 0 ;;
    [sS]) echo "  → Saltado." ; return 1 ;;
    *)    echo "  → Saltado." ; return 1 ;;
  esac
}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 0. Ir al directorio base ────────────────────────────────────────────
cd "$BASE" || { echo "ERROR: no existe $BASE"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        datahub-ocr — despliegue automático                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  • Contenedor origen: $COLAB"
echo "║  • Imagen destino:    $IMAGE"
echo "║  • GPUs asignadas:    2,3 (físicas) → 0,1 (lógicas dentro)"
echo "║  • Periódico:         El Comercio (2001-2022)"
echo "║  • Propiedad salidas: fteran:root"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Verificar pre-requisitos ──────────────────────────────────────────
log "Verificando pre-requisitos..."

ERR=0
docker ps >/dev/null 2>&1 || { echo "  ✗ docker no disponible (¿permisos?)"; ERR=1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "  ✗ nvidia-smi falló"; ERR=1; }
[ -f "$BASE/docling_suryaOCR.py" ] || { echo "  ✗ docling_suryaOCR.py no está en $BASE"; ERR=1; }
docker ps --filter name="$COLAB" --format '{{.Names}}' | grep -q "$COLAB" || {
  echo "  ✗ Contenedor $COLAB no está corriendo"
  ERR=1
}
[ -d "/home/fteran/dhub/El Comercio 2001-2022" ] || {
  echo "  ✗ No se encuentra /home/fteran/dhub/El Comercio 2001-2022"
  ERR=1
}
if [ "$ERR" -eq 1 ]; then
  echo "  Corrige los errores y vuelve a ejecutar."
  exit 1
fi
log "  ✓ docker, nvidia-smi, script, colab_fteran, datos OK"

# ── 2. Matar OCR manual dentro del contenedor colab ─────────────────────
if confirm "¿Matar procesos OCR existentes en $COLAB? (esto NO afecta salidas ya escritas)"; then
  log "Matando procesos docling_suryaOCR dentro de $COLAB..."
  docker exec "$COLAB" bash -c '
    pkill -f docling_suryaOCR.py 2>/dev/null
    sleep 3
    pkill -9 -f docling_suryaOCR.py 2>/dev/null
  ' || true
  log "  ✓ Procesos OCR eliminados (si los había)"
  sleep 2
fi

# ── 3. Commit del contenedor colab → imagen datahub-ocr ─────────────────
EXISTE=$(docker images -q "$IMAGE" 2>/dev/null)
if [ -n "$EXISTE" ]; then
  log "  La imagen $IMAGE ya existe ($(docker images --format '{{.Size}}' --filter "reference=$IMAGE"))"
  if confirm "¿Sobrescribir con un nuevo commit de $COLAB?"; then
    log "Commit de $COLAB → $IMAGE ..."
    docker commit "$COLAB" "$IMAGE"
    log "  ✓ Imagen creada: $IMAGE"
  else
    log "  Usando imagen existente."
  fi
else
  log "Commit de $COLAB → $IMAGE ..."
  docker commit "$COLAB" "$IMAGE"
  log "  ✓ Imagen creada: $IMAGE"
fi

# ── 4. Verificar que runner scripts estén ejecutables ────────────────────
chmod +x "$BASE/runner/"*.sh
log "  ✓ Scripts ejecutables"

# ── 5. Verificar que datahub_ctl.sh tenga la config correcta ────────────
log "Verificando configuración en runner/datahub_ctl.sh..."
grep -q 'DEVICES="${DEVICES:-2,3}"' "$BASE/runner/datahub_ctl.sh" && \
  log "  ✓ DEVICES=2,3"
grep -q "IN_BASE_HOST.*/home/fteran/dhub" "$BASE/runner/datahub_ctl.sh" && \
  log "  ✓ IN_BASE_HOST apunta a dhub"
grep -q "OUT_BASE_HOST.*Proyecto_DataHub" "$BASE/runner/datahub_ctl.sh" && \
  log "  ✓ OUT_BASE_HOST apunta a Proyecto_DataHub"

# ── 6. Instalar crontab ──────────────────────────────────────────────────
if confirm "¿Instalar crontab con @reboot + status + recreate diario?"; then
  log "Instalando crontab desde runner/crontab.datahub ..."
  crontab "$BASE/runner/crontab.datahub"
  log "  ✓ crontab instalado. Ver con: crontab -l"
fi

# ── 7. Matar cualquier contenedor datahub_ocr previo y lanzar ──────────
if confirm "¿Detener contenedor previo (si existe) y lanzar datahub_ocr?"; then
  log "Ejecutando: datahub_ctl.sh recreate ..."
  "$BASE/runner/datahub_ctl.sh" recreate
  sleep 3

  # Esperar un poco y verificar
  sleep 5
  echo ""
  "$BASE/runner/datahub_ctl.sh" status
fi

# ── 8. Resumen final ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              DESPLIEGUE COMPLETADO                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  • Contenedor:   datahub_ocr                                ║"
echo "║  • Status:       datahub_ctl.sh status                      ║"
echo "║  • Logs vivo:    datahub_ctl.sh logs                        ║"
echo "║  • Progreso:     python3 runner/progress.py --watch 30      ║"
echo "║  • Detener:      datahub_ctl.sh stop                        ║"
echo "║                                                             ║"
echo "║  AUTO-REINICIO ACTIVO:                                      ║"
echo "║  • Proceso:  supervisor.sh lo re-lanza (~15s)               ║"
echo "║  • Contenedor: restart=always de Docker                     ║"
echo "║  • Máquina:   @reboot cron → datahub_ctl.sh ensure          ║"
echo "║  • Diario:    22:00 → recreate (estado limpio)              ║"
echo "║                                                             ║"
echo "║  Única forma de parar:  datahub_ctl.sh stop                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
