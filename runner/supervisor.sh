#!/usr/bin/env bash
# supervisor.sh (H200) — Proceso principal del contenedor Docker (datahub_ctl.sh).
#
#  - Reparte el trabajo entre GPUs balanceando por imágenes PENDIENTES
#    (balance_folders.py, LPT a nivel mes). Cada arranque rebalancea.
#  - Rango abierto: [YEAR_MIN .. YEAR_MAX(9999)] con EXCLUDE (regex).
#  - 1 proceso por GPU lógica; el .py de la H200 usa hilos (--parallel).
#  - Re-lanza el proceso que se caiga. Reanuda: el .py salta imágenes con raw.md.
#  - Maneja SIGTERM/SIGINT para que stop/recreate sean limpios.
set -u

# ── Config por variables de entorno (los -e del docker run pisan esto) ──────
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"   # venv surya2 montado en /opt/venv
IN_BASE="${IN_BASE:-/data/in}"
OUT_BASE="${OUT_BASE:-/data/out}"
SCRIPT="${SCRIPT:-/data/out/docling_suryaOCR.py}"
LOGDIR="${LOGDIR:-$OUT_BASE/logs_corpus}"
YEAR_MIN="${YEAR_MIN:-2001}"
YEAR_MAX="${YEAR_MAX:-9999}"        # abierto hacia arriba (la H200 hace 2013+)
PARALLEL="${PARALLEL:-4}"           # hilos por proceso (el .py de la H200 usa --parallel)
RESPAWN_DELAY="${RESPAWN_DELAY:-15}"
RECHECK_DONE_EVERY="${RECHECK_DONE_EVERY:-1800}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-52428800}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
read -ra GPUS <<< "${GPUS:-0 1}"    # GPUs LÓGICAS dentro del contenedor

# Carpetas a EXCLUIR (regex, '|' separa patrones, matchea ruta completa,
# case-insensitive). '$' y '.' escapados para servir en grep -E y Python.
EXCLUDE="${EXCLUDE:-\\\$RECYCLE\\.BIN|System Volume Information}"

mkdir -p "$LOGDIR"
log() { echo "[$(date '+%F %T')][supervisor] $*"; }

log "PID $$ | GPUs lógicas: ${GPUS[*]} | años [$YEAR_MIN..$YEAR_MAX] | parallel=$PARALLEL"
log "PYTHON_BIN=$PYTHON_BIN"
log "IN_BASE=$IN_BASE | OUT_BASE=$OUT_BASE | SCRIPT=$SCRIPT"
log "EXCLUDE=$EXCLUDE"

if [ ! -x "$PYTHON_BIN" ]; then
  log "ERROR: no existe/ejecuta PYTHON_BIN=$PYTHON_BIN (¿venv montado?)."
  exit 1
fi

# ── Descubrir carpetas-año (para el fallback round-robin) ───────────────────
discover_dirs() {
  mapfile -t YEAR_DIRS < <(
    for paper in "$IN_BASE"/*/; do
      [ -d "$paper" ] || continue
      for yd in "$paper"*/; do
        yd="${yd%/}"
        [ -d "$yd" ] || continue
        if [ -n "$EXCLUDE" ] && echo "$yd" | grep -qiE "$EXCLUDE"; then
          continue
        fi
        yr=$(basename "$yd" | grep -oE '[0-9]{4}$')
        if [ -n "$yr" ] && [ "$yr" -ge "$YEAR_MIN" ] && [ "$yr" -le "$YEAR_MAX" ]; then
          echo "$yd"
        fi
      done
    done | sort
  )
}

discover_dirs
if [ "${#YEAR_DIRS[@]}" -eq 0 ]; then
  log "No encontré carpetas-año en [$YEAR_MIN..$YEAR_MAX] bajo $IN_BASE (¿volumen sin montar?). Reintento en 5 min."
  sleep 300
  exec "$0" "$@"
fi
log "Carpetas-año (tras EXCLUDE): ${#YEAR_DIRS[@]}"

# ── Reparto balanceado por imágenes PENDIENTES (LPT, nivel mes) ─────────────
RUNNER_DIR="$(cd "$(dirname "$0")" && pwd)"
bal_out="$("$PYTHON_BIN" "$RUNNER_DIR/balance_folders.py" \
      --in-base "$IN_BASE" --out-base "$OUT_BASE" \
      --year-min "$YEAR_MIN" --year-max "$YEAR_MAX" \
      --logdir "$LOGDIR" --gpus "${GPUS[*]}" \
      --exclude "$EXCLUDE" 2>&1)"; bal_rc=$?
[ -n "$bal_out" ] && echo "$bal_out" | sed 's/^/[supervisor] /'
if [ "$bal_rc" -ne 0 ]; then
  log "Balanceador falló (rc=$bal_rc); uso round-robin simple como respaldo."
  n=${#GPUS[@]}
  for gpu in "${GPUS[@]}"; do : > "$LOGDIR/folders_gpu${gpu}.txt"; done
  for idx in "${!YEAR_DIRS[@]}"; do
    gpu=${GPUS[$(( idx % n ))]}
    echo "${YEAR_DIRS[$idx]}" >> "$LOGDIR/folders_gpu${gpu}.txt"
  done
fi
for gpu in "${GPUS[@]}"; do
  log "GPU $gpu <- $(wc -l < "$LOGDIR/folders_gpu${gpu}.txt") unidades"
done

declare -A PIDS DONE

trim_log() {
  local f="$1" sz
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt "$LOG_MAX_BYTES" ]; then
    tail -n 3000 "$f" > "$f.tmp" 2>/dev/null && mv "$f.tmp" "$f"
  fi
}

launch_gpu() {
  local gpu="$1"
  local flist="$LOGDIR/folders_gpu${gpu}.txt"
  if [ ! -s "$flist" ]; then
    log "GPU $gpu sin carpetas asignadas; omito."
    DONE[$gpu]=1; PIDS[$gpu]=""
    return 0
  fi
  local logf="$LOGDIR/run_gpu${gpu}.log"
  trim_log "$logf"
  echo "===== $(date '+%F %T') :: (re)lanzando GPU $gpu =====" >> "$logf"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$SCRIPT" \
      --in-base "$IN_BASE" --out-base "$OUT_BASE" \
      --parallel "$PARALLEL" --folders-file "$flist" \
      >> "$logf" 2>&1 &
  PIDS[$gpu]=$!
  DONE[$gpu]=0
  log "GPU $gpu -> PID ${PIDS[$gpu]} | log $logf"
}

shutdown() {
  log "Señal de parada recibida; terminando procesos hijos..."
  local gpu pid
  for gpu in "${GPUS[@]}"; do
    pid="${PIDS[$gpu]:-}"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
  done
  local i alive
  for i in $(seq 1 10); do
    alive=0
    for gpu in "${GPUS[@]}"; do
      pid="${PIDS[$gpu]:-}"
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ "$alive" -eq 0 ] && break
    sleep 1
  done
  log "Hijos terminados. Saliendo (rc=0)."
  exit 0
}
trap shutdown SIGTERM SIGINT

for gpu in "${GPUS[@]}"; do launch_gpu "$gpu"; done
last_recheck=$SECONDS

while true; do
  for gpu in "${GPUS[@]}"; do
    pid="${PIDS[$gpu]:-}"
    if [ -z "$pid" ]; then
      [ "${DONE[$gpu]:-0}" -eq 1 ] || launch_gpu "$gpu"
      continue
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null; rc=$?
      if [ "$rc" -eq 0 ]; then
        log "GPU $gpu terminó OK (rc=0). En espera."
        DONE[$gpu]=1; PIDS[$gpu]=""
      else
        log "GPU $gpu CAÍDO (rc=$rc). Re-lanzo en ${RESPAWN_DELAY}s."
        sleep "$RESPAWN_DELAY"
        launch_gpu "$gpu"
      fi
    fi
  done

  if [ "$RECHECK_DONE_EVERY" -gt 0 ] && (( SECONDS - last_recheck >= RECHECK_DONE_EVERY )); then
    last_recheck=$SECONDS
    for gpu in "${GPUS[@]}"; do
      if [ "${DONE[$gpu]:-0}" -eq 1 ] && [ -s "$LOGDIR/folders_gpu${gpu}.txt" ]; then
        log "Re-escaneo GPU $gpu por si llegaron imágenes nuevas."
        launch_gpu "$gpu"
      fi
    done
  fi

  sleep 20
done