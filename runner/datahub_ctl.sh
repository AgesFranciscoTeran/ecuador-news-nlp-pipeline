#!/usr/bin/env bash
  # datahub_ctl.sh — Control del contenedor OCR en el HOST.
  #
  # Uso: datahub_ctl.sh {run|ensure|recreate|stop|status|logs}
  #   run       crea y arranca el contenedor desde cero
  #   ensure    arranca solo si NO está corriendo        (para @reboot)
  #   recreate  elimina y recrea desde cero              (para el cron de las 22:00)
  #   stop      detiene y elimina el contenedor
  #   status    docker ps + tabla de progreso
  #   logs      sigue la salida (docker logs -f)
  #
  # Cada (re)creación reanuda el trabajo: el .py salta lo ya procesado.
  set -u
  
  # ── CONFIG (edita esto) ─────────────────────────────────────────────────────
  IMAGE="${IMAGE:-datahub-ocr:latest}"        # imagen con docling + surya + torch ya instalados
  NAME="${NAME:-datahub_ocr}"
  DEVICES="${DEVICES:-2,3}"     # 2 GPUs físicas (2 y 3)              # GPUs FÍSICAS asignadas (las 2 de ID más alto)
  IN_BASE_HOST="${IN_BASE_HOST:-/home/fteran/dhub}"
  OUT_BASE_HOST="${OUT_BASE_HOST:-/home/fteran/Proyecto_DataHub}"
  CACHE_HOST="${CACHE_HOST:-/home/fteran/Proyecto_DataHub/model_cache}"
  YEAR_MIN="${YEAR_MIN:-2001}"
  PARALLEL="${PARALLEL:-4}"
  SHM="${SHM:-8g}"
  # ────────────────────────────────────────────────────────────────────────────
  # NOTA: GPUS="0 1" (abajo) es CORRECTO y NO se toca. Docker expone solo las GPUs
  # físicas de DEVICES y las renumera como 0,1 dentro del contenedor.
  
  mkdir -p "$OUT_BASE_HOST/logs_corpus" "$CACHE_HOST"
  
_remove() {
  # Procesos GPU atascados (estado D) pueden demorar el stop varios minutos.
  # Reintentamos stop+rm hasta ~10 min en vez de rendirnos al primer fallo:
  # un solo rm fallido + espera pasiva NUNCA libera el nombre (el contenedor
  # queda registrado como Exited). Hay que volver a intentar el rm.
  local i
  for i in $(seq 1 15); do
    docker ps -a --format '{{.Names}}' | grep -qx "$NAME" || return 0
    docker stop -t 30 "$NAME" 2>&1 | sed 's/^/  [stop] /'
    docker rm "$NAME"         2>&1 | sed 's/^/  [rm]   /'
    docker ps -a --format '{{.Names}}' | grep -qx "$NAME" || return 0
    echo "  [retry $i/15] '$NAME' aún registrado; reintento en 30s..."
    sleep 30
  done
  echo "  [ERROR] '$NAME' sigue registrado tras ~10 min; no recreo para no chocar."
  return 1
}

  _run() {
    docker run -d \
      --name "$NAME" \
      --restart=always \
      --init \
      --gpus "\"device=${DEVICES}\"" \
      --shm-size="$SHM" \
      --user "$(id -u fteran):$(id -g fteran)" \
      -e GPUS="0 1" \
      -e YEAR_MIN="$YEAR_MIN" \
      -e PARALLEL="$PARALLEL" \
      -e IN_BASE=/data/in \
      -e OUT_BASE=/data/out \
      -e LOGDIR=/data/out/logs_corpus \
      -e SCRIPT=/data/out/docling_suryaOCR.py \
      -e HOME=/opt/cache \
      -e MPLCONFIGDIR=/opt/cache/mpl \
      -e HF_HOME=/opt/cache/hf \
      -e HUGGINGFACE_HUB_CACHE=/opt/cache/hf \
      -e TORCH_HOME=/opt/cache/torch \
      -e XDG_CACHE_HOME=/opt/cache \
      -v "$IN_BASE_HOST":/data/in:ro \
      -v "$OUT_BASE_HOST":/data/out \
      -v "$CACHE_HOST":/opt/cache \
      -v "/home/fteran/Proyecto_DataHub/venvs/surya2":/opt/venv \
      --entrypoint bash \
      "$IMAGE" \
      /data/out/runner/supervisor.sh
  }
  
  _running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]
  }
  
  cmd="${1:-status}"
  case "$cmd" in
    run)
      if _running; then echo "'$NAME' ya está corriendo."; else _run && echo "'$NAME' arrancado."; fi
      ;;
    ensure)
      if _running; then
        echo "[ensure $(date '+%F %T')] '$NAME' ya corriendo; nada que hacer."
      else
        _remove || exit 1
        _run && echo "[ensure $(date '+%F %T')] '$NAME' arrancado."
      fi
      ;;
    recreate)
      echo "[recreate $(date '+%F %T')] eliminando y recreando '$NAME'..."
      _remove || exit 1
      _run && echo "[recreate $(date '+%F %T')] '$NAME' recreado y corriendo."
      ;;
    stop)
      docker rm -f "$NAME" >/dev/null 2>&1 && echo "Detenido y eliminado '$NAME'." || echo "No estaba corriendo."
      ;;
    status)
      docker ps --filter "name=^/${NAME}$" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}' || true
      echo
      python3 "$OUT_BASE_HOST/runner/progress.py" --logdir "$OUT_BASE_HOST/logs_corpus" 2>/dev/null || true
      ;;
    logs)
      docker logs -f "$NAME"
      ;;
    *)
      echo "Uso: $0 {run|ensure|recreate|stop|status|logs}"; exit 1
      ;;
  esac