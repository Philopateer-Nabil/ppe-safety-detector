"""Export a YOLOv8 model to ONNX format for optimised inference.

Usage::

    python export_onnx.py                        # uses default yolov8n.pt
    python export_onnx.py --weights best.pt      # use a custom-trained checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import MODEL_DIR, YOLO_WEIGHTS


def export(weights: Path, imgsz: int = 640, opset: int = 12) -> Path:
    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)

    onnx_path = weights.with_suffix(".onnx")
    dest = MODEL_DIR / onnx_path.name
    if onnx_path != dest:
        onnx_path.rename(dest)

    print(f"Exported ONNX model to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument(
        "--weights",
        type=Path,
        default=YOLO_WEIGHTS,
        help="Path to .pt weights file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    export(args.weights, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
