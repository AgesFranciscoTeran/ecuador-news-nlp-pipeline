#!/usr/bin/env python3
"""
balance_folders.py — Reparte carpetas entre GPUs balanceando por imágenes
PENDIENTES (las que aún no tienen raw.md), no por número de carpetas.

- Unidades a nivel MES (subcarpetas del año) para granularidad fina;
  si un año no tiene subcarpetas, el año entero es la unidad.
- Asignación LPT (Longest Processing Time): la unidad más pesada va al
  bin (GPU) con menos pendientes acumulados.
- Escribe LOGDIR/folders_gpu<ID>.txt (mismo formato que consume el .py).
- Se corre en cada arranque del supervisor -> cada reinicio REBALANCEA
  contra lo que queda, así el desbalance no se acumula.

Uso:
  balance_folders.py --in-base ... --out-base ... --year-min 2006 \
      --year-max 2013 --logdir ... --gpus "0 1 2 3"
"""
import argparse
import re
import sys
from pathlib import Path

EXCLUDE_DEFAULT = r"\$RECYCLE\.BIN|System Volume Information"


def find_year_dirs(in_base: Path, ymin: int, ymax: int, exclude_re):
    for paper in sorted(p for p in in_base.iterdir() if p.is_dir()):
        for yd in sorted(q for q in paper.iterdir() if q.is_dir()):
            if exclude_re and exclude_re.search(str(yd)):
                continue
            m = re.search(r"(\d{4})$", yd.name)
            if m and ymin <= int(m.group(1)) <= ymax:
                yield yd


def units_for(year_dir: Path):
    subs = sorted(q for q in year_dir.iterdir() if q.is_dir())
    return subs if subs else [year_dir]


def pending_count(unit: Path, in_base: Path, out_base: Path) -> int:
    n = 0
    for pat in ("*.jpg", "*.JPG"):
        for img in unit.rglob(pat):
            raw = out_base / img.relative_to(in_base).parent / img.stem / "raw.md"
            if not raw.exists():
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Balanceador LPT por imágenes pendientes")
    ap.add_argument("--in-base", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--year-min", type=int, required=True)
    ap.add_argument("--year-max", type=int, required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--gpus", required=True, help='IDs separados por espacio, ej. "0 1 2 3"')
    ap.add_argument("--exclude", default=EXCLUDE_DEFAULT,
                    help="regex de rutas a excluir (default: basura de Windows)")
    args = ap.parse_args()

    in_base = Path(args.in_base)
    out_base = Path(args.out_base)
    logdir = Path(args.logdir)
    gpus = args.gpus.split()
    exclude_re = re.compile(args.exclude, re.IGNORECASE) if args.exclude else None

    year_dirs = list(find_year_dirs(in_base, args.year_min, args.year_max, exclude_re))
    if not year_dirs:
        print(f"[balance] No hay carpetas-año en [{args.year_min}..{args.year_max}] bajo {in_base}",
              file=sys.stderr)
        return 2

    # Unidades (nivel mes) + conteo de pendientes por unidad
    units = []
    for yd in year_dirs:
        for u in units_for(yd):
            units.append((u, pending_count(u, in_base, out_base)))

    total = sum(c for _, c in units)
    print(f"[balance] {len(year_dirs)} años -> {len(units)} unidades | pendientes totales: {total}")

    # LPT: más pesada primero, al bin con menos carga
    units.sort(key=lambda t: t[1], reverse=True)
    bins = {g: {"count": 0, "units": []} for g in gpus}
    for u, c in units:
        g = min(gpus, key=lambda g: bins[g]["count"])
        bins[g]["count"] += c
        bins[g]["units"].append(u)

    logdir.mkdir(parents=True, exist_ok=True)
    for g in gpus:
        # ordena las unidades del bin para lectura estable/reproducible
        paths = sorted(str(u) for u in bins[g]["units"])
        (logdir / f"folders_gpu{g}.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
        print(f"[balance] GPU {g}: {len(paths):>4} unidades | {bins[g]['count']:>8} pendientes")

    if total:
        counts = [bins[g]["count"] for g in gpus]
        spread = (max(counts) - min(counts)) / (total / len(gpus)) * 100 if total else 0
        print(f"[balance] desbalance máx-mín: {max(counts)-min(counts)} imgs "
              f"({spread:.1f}% del promedio por GPU)")
    return 0


if __name__ == "__main__":
    sys.exit(main())