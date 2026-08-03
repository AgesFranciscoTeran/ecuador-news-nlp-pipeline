#!/usr/bin/env python3
"""
docling_suryaOCR.py — OCR de producción (Docling + Surya) sobre imágenes de
prensa escaneada.

Diseñado para corridas de días bajo supervisor.sh:
  * Idempotente: salta toda imagen que ya tenga raw.md, así el respawn del
    supervisor y el `recreate` diario reanudan en vez de reprocesar.
  * Tolerante a fallos: cada error se escribe en ERROR.txt junto a la imagen y
    el proceso continúa; una página corrupta no tumba la corrida.
  * Consultable en vivo: `kill -USR1 <pid>` imprime avance sin detener nada.
  * Un converter por hilo, para evitar condiciones de carrera en el pool.

Salida por imagen, espejando la ruta relativa de entrada:
  <out-base>/<ruta relativa>/<nombre sin extensión>/raw.md
  <out-base>/<ruta relativa>/<nombre sin extensión>/docling.json
"""
import os
import json
import sys
import time
import signal
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import PIL.ImageFile
import torch

# Tolera JPGs truncados (bytes faltantes al final del archivo)
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Parches de compatibilidad surya-ocr 0.17.1 + transformers >= 4.51 ───────
#
# surya-ocr 0.17.1 se escribió contra una API de transformers anterior a 4.51.
# Estos parches la reconstruyen. Son la parte más frágil del script: si alguno
# falla, el síntoma aparece mucho después disfrazado de error del modelo, así
# que cada bloque REPORTA su resultado en vez de callarlo.
_PATCHES: list[tuple[str, bool, str]] = []


def _patch(nombre: str):
    """Decorador: ejecuta el parche y registra si se aplicó o por qué falló."""
    def deco(fn):
        try:
            fn()
            _PATCHES.append((nombre, True, ""))
        except Exception as e:                                    # noqa: BLE001
            _PATCHES.append((nombre, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


@_patch("SuryaDecoderConfig.pad_token_id")
def _p1():
    from surya.common.surya.decoder.config import SuryaDecoderConfig
    SuryaDecoderConfig.pad_token_id = None


@_patch("ROPE_INIT_FUNCTIONS['default']")
def _p2():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _RIF
    if "default" in _RIF:
        return
    def _compute_default_rope_parameters(
        config=None, device=None, seq_len=None, layer_type=None,
    ):
        config.standardize_rope_params()
        d = (config.rope_parameters[layer_type] if layer_type is not None
             else config.rope_parameters)
        factor = d.get("factor", 1.0)
        base = d["rope_theta"]
        head_dim = (getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads)
        p_rot = d.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * p_rot)
        inv_freq = 1.0 / (base ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        inv_freq /= factor
        return inv_freq, 1.0
    _RIF["default"] = _compute_default_rope_parameters


@_patch("SuryaModel.all_tied_weights_keys")
def _p3():
    from surya.common.surya import SuryaModel
    # transformers >= 4.51 espera all_tied_weights_keys (lo fija post_init(),
    # que surya-ocr 0.17.1 nunca llama).
    _orig_init = SuryaModel.__init__

    def _surya_model_init(self, config, *args, **kwargs):
        _orig_init(self, config, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
    SuryaModel.__init__ = _surya_model_init


@_patch("SuryaModel.tie_weights(**kwargs)")
def _p4():
    from surya.common.surya import SuryaModel
    # transformers >= 4.51 pasa kwargs a tie_weights(); surya espera solo self.
    _orig_tie = SuryaModel.tie_weights

    def _surya_tie_weights(self, **kwargs):
        return _orig_tie(self)
    SuryaModel.tie_weights = _surya_tie_weights


@_patch("SuryaModel._tie_or_clone_weights")
def _p5():
    from surya.common.surya import SuryaModel
    # transformers >= 4.51 eliminó _tie_or_clone_weights; se reimplementa.
    def _tie_or_clone_weights(output_embeddings, input_embeddings):
        output_embeddings.weight = input_embeddings.weight
    SuryaModel._tie_or_clone_weights = staticmethod(_tie_or_clone_weights)


@_patch("Qwen2_5_VisionRotaryEmbedding.forward (meta device)")
def _p6():
    from surya.common.surya.encoder import Qwen2_5_VisionRotaryEmbedding
    # inv_freq se crea como tensor plano (no buffer), así que se queda en meta
    # device; se recrea en CPU dentro del forward.
    _orig_vfwd = Qwen2_5_VisionRotaryEmbedding.forward

    def _vfwd(self, seqlen):
        if self.inv_freq.device.type == "meta":
            dim = self.inv_freq.shape[0] * 2
            self.inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return _orig_vfwd(self, seqlen)
    Qwen2_5_VisionRotaryEmbedding.forward = _vfwd


def reportar_parches() -> None:
    fallidos = [(n, err) for n, ok, err in _PATCHES if not ok]
    print(f"[parches] {len(_PATCHES) - len(fallidos)}/{len(_PATCHES)} aplicados")
    for nombre, err in fallidos:
        print(f"[parches][AVISO] NO se aplicó '{nombre}' -> {err}", file=sys.stderr)
    if fallidos:
        print("[parches][AVISO] Si el OCR falla más adelante, empieza por aquí: "
              "puede ser una versión de transformers/surya distinta de la "
              "esperada (surya-ocr 0.17.1 + transformers 4.51.3).",
              file=sys.stderr, flush=True)


from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)

from docling_surya import SuryaOcrOptions


# ── Estado para consulta en vivo (SIGUSR1) ──────────────────────────
_done = 0
_done_lock = threading.Lock()
_t_run = None


def _bump():
    global _done
    with _done_lock:
        _done += 1


def _status_handler(signum, frame):
    n = _done
    if _t_run is None:
        print("\n[STATUS] aún no arranca el procesamiento", flush=True)
        return
    elapsed = time.perf_counter() - _t_run
    avg = elapsed / n if n else 0.0
    rate = n / elapsed if elapsed else 0.0
    print(
        f"\n[STATUS] procesadas={n} | elapsed={elapsed:.1f}s "
        f"| avg={avg:.2f}s/img | rate={rate:.2f} img/s",
        flush=True,
    )


signal.signal(signal.SIGUSR1, _status_handler)


def get_accelerator() -> AcceleratorOptions:
    if torch.cuda.is_available():
        device = AcceleratorDevice.CUDA
    elif torch.backends.mps.is_available():
        device = AcceleratorDevice.MPS
    else:
        device = AcceleratorDevice.CPU
    return AcceleratorOptions(device=device)


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="suryaocr",
        allow_external_plugins=True,
        ocr_options=SuryaOcrOptions(lang=["es"]),
        accelerator_options=get_accelerator(),
    )
    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
        }
    )


# Cada hilo del pool tiene su propio converter -> sin condiciones de carrera.
_thread_local = threading.local()


def _get_converter() -> DocumentConverter:
    conv = getattr(_thread_local, "converter", None)
    if conv is None:
        conv = _build_converter()
        _thread_local.converter = conv
    return conv


def out_dir_for(img: Path, in_base: Path, out_base: Path) -> Path:
    # Espeja la ruta relativa a in_base bajo out_base
    return out_base / img.relative_to(in_base).parent / img.stem


def already_done(img: Path, in_base: Path, out_base: Path) -> bool:
    return (out_dir_for(img, in_base, out_base) / "raw.md").exists()


def save_result(img: Path, result, in_base: Path, out_base: Path) -> str:
    out_dir = out_dir_for(img, in_base, out_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.status != ConversionStatus.SUCCESS:
        (out_dir / "ERROR.txt").write_text(str(result.errors), encoding="utf-8")
        return "error"

    try:
        raw_md = result.document.export_to_markdown()
        (out_dir / "raw.md").write_text(raw_md, encoding="utf-8")
        _bump()
    except Exception as e:                                        # noqa: BLE001
        (out_dir / "ERROR.txt").write_text(str(e), encoding="utf-8")
        return "error"

    try:
        doc_dict = result.document.export_to_dict()
        (out_dir / "docling.json").write_text(
            json.dumps(doc_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:                                        # noqa: BLE001
        (out_dir / "docling_json_error.txt").write_text(str(e), encoding="utf-8")

    return "ok"


def process_one(img: Path, in_base: Path, out_base: Path) -> str:
    conv = _get_converter()
    try:
        result = conv.convert(str(img), raises_on_error=False)
    except Exception as e:                                        # noqa: BLE001
        out_dir = out_dir_for(img, in_base, out_base)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ERROR.txt").write_text(str(e), encoding="utf-8")
        return "error"
    return save_result(img, result, in_base, out_base)


def collect_roots(args) -> list[Path]:
    roots = [Path(p) for p in (args.folders or [])]
    if args.folders_file:
        for line in Path(args.folders_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                roots.append(Path(line))
    return roots


def collect_images(roots: list[Path], exts: set[str]) -> tuple[list[Path], dict]:
    """Recorre cada raíz una sola vez y filtra por extensión sin distinguir
    mayúsculas, evitando duplicados si dos raíces se solapan. Devuelve las
    imágenes y el conteo por extensión, para detectar formatos inesperados."""
    vistos: set[str] = set()
    imagenes: list[Path] = []
    por_ext: dict[str, int] = {}
    for r in roots:
        for p in sorted(r.rglob("*")):
            suf = p.suffix.lower()
            if suf not in exts or not p.is_file():
                continue
            clave = str(p)
            if clave in vistos:
                continue
            vistos.add(clave)
            imagenes.append(p)
            por_ext[suf] = por_ext.get(suf, 0) + 1
    return imagenes, por_ext


def main():
    global _t_run

    parser = argparse.ArgumentParser(description="OCR producción Surya/Docling")
    parser.add_argument("--in-base", required=True,
                        help="Raíz común de entrada (ej. /home/fteran/dhub)")
    parser.add_argument("--out-base", required=True,
                        help="Raíz común de salida (ej. /home/fteran/Proyecto_DataHub)")
    parser.add_argument("--folders", nargs="*", default=[],
                        help="Carpetas a procesar (se recorren recursivamente)")
    parser.add_argument("--folders-file", default=None,
                        help="Archivo con una carpeta por línea (maneja espacios)")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Imágenes concurrentes por GPU (default 4)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="RECOGNITION_BATCH_SIZE de surya. 0 = default (256).")
    parser.add_argument("--ext", default="jpg,jpeg,png,tif,tiff",
                        help="Extensiones a procesar, separadas por coma. "
                             "No distingue mayúsculas (default: jpg,jpeg,png,tif,tiff)")
    args = parser.parse_args()

    in_base = Path(args.in_base)
    out_base = Path(args.out_base)
    exts = {"." + e.strip().lower().lstrip(".")
            for e in args.ext.split(",") if e.strip()}

    if args.batch_size > 0:
        os.environ["RECOGNITION_BATCH_SIZE"] = str(args.batch_size)
        os.environ["DETECTOR_BATCH_SIZE"] = str(max(1, args.batch_size // 8))

    print(f"[pid] {os.getpid()}  (consulta con: kill -USR1 {os.getpid()})")
    print(f"[config] parallel={args.parallel} | batch={args.batch_size or 'default(256)'} "
          f"| ext={','.join(sorted(e.lstrip('.') for e in exts))}")
    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[device] {dev}  | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(no set)')}")
    reportar_parches()

    roots = collect_roots(args)
    if not roots:
        print("No me diste carpetas (--folders o --folders-file).")
        return
    print(f"[roots] {len(roots)} carpetas:")
    for r in roots:
        print(f"   - {r}")

    images, por_ext = collect_images(roots, exts)
    detalle = " | ".join(f"{k} {v}" for k, v in sorted(por_ext.items()))
    print(f"Found {len(images)} images  ({detalle or 'ninguna'})")

    pending = [img for img in images if not already_done(img, in_base, out_base)]
    skipped = len(images) - len(pending)
    print(f"  Skip (ya procesadas): {skipped}  |  Pending: {len(pending)}")
    if not pending:
        print("Nada que procesar.")
        return

    t_run = time.perf_counter()
    _t_run = t_run
    ok_count = err_count = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(process_one, img, in_base, out_base): img
            for img in pending
        }
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            if status == "ok":
                ok_count += 1
            else:
                err_count += 1
            elapsed = time.perf_counter() - t_run
            avg = elapsed / i
            eta = avg * (len(pending) - i)
            print(
                f"[{i:>5}/{len(pending)}] {status} | avg {avg:.1f}s/img | ETA {eta/60:.1f}min",
                flush=True,
            )

    t_elapsed = time.perf_counter() - t_run
    rate = ok_count / t_elapsed if t_elapsed else 0.0
    print("\n── Resumen ─────────────────────────────────────────")
    print(f"  Procesadas : {ok_count}")
    print(f"  Errores    : {err_count}")
    print(f"  Tiempo     : {t_elapsed:.1f}s  ({t_elapsed/max(ok_count,1):.2f}s/img)")
    print(f"  Rate       : {rate:.2f} img/s")


if __name__ == "__main__":
    main()