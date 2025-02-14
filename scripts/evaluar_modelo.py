#!/usr/bin/env python3
"""Corre el modelo contra un dataset anotado en formato YOLO (imagen.jpg +
imagen.txt) y te tira precisión, recall, F1 y un CSV con el IoU por imagen.

Ejemplo:
    python scripts/evaluar_modelo.py --dataset ruta/a/darknet/data/obj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deteccion_armas import WeaponDetector  # noqa: E402
from deteccion_armas.evaluar import evaluate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Carpeta con pares imagen.jpg + imagen.txt")
    parser.add_argument("--pesos", default="models/yolov3_custom_last.weights")
    parser.add_argument("--config", default="models/yolov3_custom.cfg")
    parser.add_argument("--confianza", type=float, default=0.5)
    parser.add_argument("--iou-umbral", type=float, default=0.5)
    parser.add_argument("--salida-csv", default="results/iou_por_imagen.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = WeaponDetector(args.pesos, args.config, confidence_threshold=args.confianza)

    result = evaluate_dataset(
        detector,
        images_dir=args.dataset,
        iou_threshold=args.iou_umbral,
        output_csv=args.salida_csv,
    )

    print(result.summary())
    print(f"IoU por imagen guardado en: {args.salida_csv}")


if __name__ == "__main__":
    main()
