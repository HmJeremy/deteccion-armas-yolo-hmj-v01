"""Detector de armas de fuego sobre YOLOv3/v4 (OpenCV DNN).

Es un wrapper de ``cv2.dnn`` para cargar un modelo Darknet (.cfg + .weights)
y correr inferencia sobre imágenes o frames de video. A diferencia de los
scripts originales de la tesis (que mandaban todas las detecciones con
confianza > 0 directo a NMS, sin filtrar nada), acá sí se aplica un umbral
de confianza de verdad antes de NMS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Una detección: caja delimitadora + clase + confianza."""

    x: int
    y: int
    w: int
    h: int
    class_id: int
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class WeaponDetector:
    """Carga un modelo YOLO entrenado para detectar armas y corre inferencia.

    Parameters
    ----------
    weights_path, config_path:
        Rutas al ``.weights`` y ``.cfg`` de Darknet.
    classes:
        Nombres de las clases, en el mismo orden que en el entrenamiento.
        Por defecto ``["Arma"]``, que es el modelo de una sola clase de la
        tesis.
    input_size:
        Tamaño (ancho, alto) del blob de entrada. Lo normal en YOLOv3/v4
        es 416.
    confidence_threshold:
        Confianza mínima para que una detección cuente. Los scripts
        originales no filtraban nada (usaban 0); acá sí se filtra antes de
        NMS.
    nms_threshold:
        Umbral de IoU para non-maxima suppression.
    backend, target:
        Backend/target de ``cv2.dnn`` (CPU por defecto). Si compilaste
        OpenCV con soporte CUDA, pásale ``cv2.dnn.DNN_BACKEND_CUDA`` /
        ``cv2.dnn.DNN_TARGET_CUDA`` para que la inferencia vaya más rápido.
    """

    def __init__(
        self,
        weights_path: str | Path,
        config_path: str | Path,
        classes: list[str] | None = None,
        input_size: tuple[int, int] = (416, 416),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        backend: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        weights_path, config_path = Path(weights_path), Path(config_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de pesos: {weights_path}"
            )
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

        self.classes = classes or ["Arma"]
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.net = cv2.dnn.readNet(str(weights_path), str(config_path))
        self.net.setPreferableBackend(backend)
        self.net.setPreferableTarget(target)
        self._output_layers = self.net.getUnconnectedOutLayersNames()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Corre el detector sobre una imagen BGR (formato OpenCV) y devuelve
        las detecciones que sobreviven al filtro de confianza + NMS."""
        height, width = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self._output_layers)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for row in output:
                scores = row[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.confidence_threshold:
                    continue

                center_x, center_y, w, h = (
                    row[0:4] * np.array([width, height, width, height])
                ).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        keep = np.array(keep).flatten() if len(keep) else []

        return [
            Detection(*boxes[i], class_id=class_ids[i], confidence=confidences[i])
            for i in keep
        ]

    def draw(
        self,
        image: np.ndarray,
        detections: list[Detection],
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Dibuja las detecciones sobre una copia de la imagen y la devuelve."""
        annotated = image.copy()
        for det in detections:
            label = self.classes[det.class_id] if det.class_id < len(self.classes) else str(det.class_id)
            text = f"{label} {det.confidence * 100:.1f}%"

            cv2.rectangle(annotated, (det.x, det.y), (det.x + det.w, det.y + det.h), color, thickness)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(
                annotated,
                (det.x, det.y + det.h),
                (det.x + tw + 6, det.y + det.h + th + 10),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                text,
                (det.x + 3, det.y + det.h + th + 4),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        return annotated

    def detect_and_draw(self, image: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        detections = self.detect(image)
        return self.draw(image, detections), detections
