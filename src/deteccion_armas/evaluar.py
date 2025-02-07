"""Evaluación del detector contra un set de anotaciones en formato YOLO.

Reemplaza los tres scripts sueltos ``pruebas_map.py`` / ``pruebas_map2.py`` /
``pruebas_map3.py`` de la versión original de la tesis (cada uno era una
variación incompleta del mismo cálculo) por una sola función reutilizable
que calcula IoU, matriz de confusión, precisión, recall y exactitud.

Espera un directorio con pares ``imagen.jpg`` + ``imagen.txt`` en formato
YOLO (``class x_center y_center width height``, normalizado 0-1), que es el
mismo formato que ya usa ``darknet/data/obj``.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .detector import WeaponDetector


def _yolo_to_xyxy(x_center: float, y_center: float, w: float, h: float, width: int, height: int) -> tuple[int, int, int, int]:
    x_min = int((x_center - w / 2) * width)
    y_min = int((y_center - h / 2) * height)
    x_max = int((x_center + w / 2) * width)
    y_max = int((y_center + h / 2) * height)
    return x_min, y_min, x_max, y_max


def intersection_over_union(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])

    inter_w, inter_h = max(0, xb - xa), max(0, yb - ya)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class EvalResult:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_image_iou: list[tuple[str, float]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        return (
            f"TP={self.true_positives}  FP={self.false_positives}  FN={self.false_negatives}\n"
            f"Precision={self.precision:.4f}  Recall={self.recall:.4f}  F1={self.f1:.4f}"
        )


def evaluate_dataset(
    detector: WeaponDetector,
    images_dir: str | Path,
    iou_threshold: float = 0.5,
    output_csv: str | Path | None = None,
) -> EvalResult:
    """Corre el detector sobre todas las imágenes de ``images_dir`` y las
    compara contra su anotación ``.txt`` (formato YOLO) usando emparejamiento
    por IoU. Devuelve un :class:`EvalResult` con TP/FP/FN agregados."""

    images_dir = Path(images_dir)
    result = EvalResult()

    for image_path in sorted(glob.glob(str(images_dir / "*.jpg"))):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = images_dir / f"{image_name}.txt"

        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        predictions = [det.box for det in detector.detect(image)]
        predictions_xyxy = [(x, y, x + w, y + h) for (x, y, w, h) in predictions]

        ground_truths = []
        if annotation_path.exists():
            with open(annotation_path, encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    _, xc, yc, w, h = parts
                    ground_truths.append(
                        _yolo_to_xyxy(float(xc), float(yc), float(w), float(h), width, height)
                    )

        matched_gt = set()
        for pred_box in predictions_xyxy:
            best_iou, best_gt_idx = 0.0, None
            for i, gt_box in enumerate(ground_truths):
                if i in matched_gt:
                    continue
                iou = intersection_over_union(gt_box, pred_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i

            result.per_image_iou.append((image_name, best_iou))
            if best_iou >= iou_threshold and best_gt_idx is not None:
                matched_gt.add(best_gt_idx)
                result.true_positives += 1
            else:
                result.false_positives += 1

        result.false_negatives += len(ground_truths) - len(matched_gt)

    if output_csv is not None:
        with open(output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["imagen", "iou"])
            writer.writerows(result.per_image_iou)

    return result
