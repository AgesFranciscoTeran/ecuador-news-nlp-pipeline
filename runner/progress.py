import os
import re
import glob
import time
import argparse
from pathlib import Path

LINE = re.compile(
    r'\[\s*(\d+)/(\d+)\]\s+\w+\s+\|\s+avg\s+([\d.]+)s/img\s+\|\s+ETA\s+([\d.]+)min'
)

IDLE_MARKERS = ("En espera", "Nada que procesar", "── Resumen", "terminó OK")


def last_progress(path, tail_bytes=65536):
    """Lee solo el final del log (rápido aunque el log sea enorme)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            data = f.read().decode("utf-8", "ignore")
    except FileNotFoundError:
        return None
    idle = any(m in data[-1500:] for m in IDLE_MARKERS)
    matches = LINE.findall(data)
    if not matches:
        return {"started": False, "idle": idle}
    i, M, avg, eta = matches[-1]
    return {
        "started": True,
        "done": int(i),
        "pending": int(M),
        "avg": float(avg),
        "eta_min": float(eta),
        "idle": idle,
    }


def fmt_eta(minutes):
    h = minutes / 60
    if h >= 48:
        return f"{h/24:.1f}d"
    return f"{h:.1f}h"


def fmt_age(seconds):
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{seconds/60:.0f}m"
    return f"{seconds/3600:.1f}h"


def render(logdir, stale_min):
    paths = sorted(glob.glob(str(Path(logdir) / "run_gpu*.log")))
    if not paths:
        print(f"  No encuentro run_gpu*.log en {logdir}")
        return

    now = time.time()
    print(f"  {'GPU':>3} | {'procesadas':>14} | {'s/img':>7} | {'img/s':>6} "
          f"| {'ETA':>6} | {'updated':>8} | estado")
    print("  " + "-" * 74)

    tot_done = tot_pending = 0
    comb_rate = 0.0
    max_eta = 0.0
    active = 0
    any_stale = False

    for p in paths:
        m = re.search(r"run_gpu(\d+)\.log", p)
        gpu = m.group(1) if m else "?"
        try:
            age = now - os.path.getmtime(p)
        except OSError:
            age = None
        d = last_progress(p)

        if d is None or not d["started"]:
            state = "✓ en espera" if (d and d.get("idle")) else "… cargando"
            age_s = fmt_age(age) if age is not None else "—"
            print(f"  {gpu:>3} | {'—':>14} | {'—':>7} | {'—':>6} | {'—':>6} "
                  f"| {age_s:>8} | {state}")
            continue

        rate = 1 / d["avg"] if d["avg"] else 0.0
        comb_rate += rate
        tot_done += d["done"]
        tot_pending += d["pending"]
        max_eta = max(max_eta, d["eta_min"])
        active += 1

        if d.get("idle"):
            state = "✓ al día"
        elif age is not None and age > stale_min * 60:
            state = f"⚠ STALE ({fmt_age(age)})"
            any_stale = True
        else:
            state = "▶ activo"

        age_s = fmt_age(age) if age is not None else "—"
        print(
            f"  {gpu:>3} | {d['done']:>5}/{d['pending']:<8} "
            f"| {d['avg']:>5.1f}s | {rate:>5.3f} | {fmt_eta(d['eta_min']):>6} "
            f"| {age_s:>8} | {state}"
        )

    print("  " + "-" * 74)
    eff = 1 / comb_rate if comb_rate else 0.0
    print(
        f"  {'TOT':>3} | {tot_done:>5}/{tot_pending:<8} "
        f"| {eff:>5.1f}s | {comb_rate:>5.3f} | {fmt_eta(max_eta):>6} "
        f"| {'':>8} | {active} GPUs activas"
    )
    if active:
        print(
            f"        s/img efectivo = {eff:.2f}  |  "
            f"throughput combinado = {comb_rate:.3f} img/s "
            f"({comb_rate*3600:.0f} img/h)"
        )
    if any_stale:
        print(f"  ⚠  Hay GPUs sin actualizar hace >{stale_min} min: posible cuelgue. "
              f"Revisa el log o fuerza:  datahub_ctl.sh recreate")


def main():
    ap = argparse.ArgumentParser(description="Tabla de progreso del OCR multi-GPU")
    ap.add_argument("--logdir", default="/home/fteran/Proyecto_DataHub/logs_corpus")
    ap.add_argument("--watch", type=int, default=0,
                    help="refrescar cada N segundos (0 = una sola vez)")
    ap.add_argument("--stale-min", type=int, default=15,
                    help="minutos sin actualizar el log para marcar una GPU como ⚠ STALE")
    args = ap.parse_args()

    while True:
        print(f"\n===== {time.strftime('%F %T')} =====")
        render(args.logdir, args.stale_min)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
