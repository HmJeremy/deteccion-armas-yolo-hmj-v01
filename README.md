# Detección Anticipada de Armas de Fuego con YOLO

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](requirements.txt)
[![OpenCV DNN](https://img.shields.io/badge/inference-OpenCV%20DNN-green)](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)

Sistema de detección de armas de fuego en tiempo real a partir de video de
cámaras de vigilancia, usando una red neuronal convolucional YOLOv3
entrenada con **Darknet** sobre un dataset propio, y corrida en producción
con el módulo `cv2.dnn` de OpenCV.

Este repositorio contiene el código de inferencia y evaluación de **"Implementación de algoritmo de redes neuronales
convolucionales para la identificación anticipada de armas de fuego"**
(UNMSM - CC - 2025).

<p align="center">
  <img src="results/samples/resultado_05.jpg" width="410" alt="Detección de arma parcialmente oculta">
  <img src="results/samples/resultado_20.jpg" width="410" alt="Detección de réplica de arma">
</p>


## Cómo funciona

```
imagen / frame de video
        │
        ▼
 blob 416×416 (cv2.dnn.blobFromImage)
        │
        ▼
  forward pass YOLOv3 (cv2.dnn.readNet)
        │
        ▼
filtrar por confianza ≥ umbral ──► Non-Maxima Suppression ──► cajas finales
```

- **`src/deteccion_armas/detector.py`** — clase `WeaponDetector`: carga el
  modelo una sola vez y expone `detect()` (devuelve las cajas) y `draw()`
  (las dibuja sobre la imagen).
- **`src/deteccion_armas/evaluar.py`** — cálculo de IoU, matriz de
  confusión, precisión/recall/F1 contra un dataset anotado en formato YOLO.
- **`main.py`** — punto de entrada único por línea de comandos: detecta
  armas en una imagen, un video, la webcam o una cámara de seguridad
  IP/RTSP, todo con el mismo comando (ver [Uso](#uso)).
- **`scripts/evaluar_modelo.py`** — evalúa el modelo contra un dataset
  anotado en formato YOLO (precisión, recall, F1, IoU por imagen).
- **`demos/`** — dos scripts didácticos usados en la sustentación (ASCII-art
  de la webcam en vivo, y visualización de los canales BGR de una imagen)
  para explicar de forma intuitiva cómo una computadora "ve" una imagen.
  No son parte del pipeline de detección.

## Instalación

```bash
git clone <url-de-este-repo>
cd deteccion-armas-yolo-hmj-v01
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Después descarga los pesos entrenados — ver
[`models/README.md`](models/README.md), no vienen en el repo por su tamaño
(~235 MB, por encima del límite de GitHub).

## Uso

`main.py` es el único punto de entrada: detecta el tipo de fuente
automáticamente (imagen fija vs. video/stream en vivo) y aplica el flujo
correspondiente.

**Detectar en una imagen:**

```bash
python main.py --fuente ejemplo.jpg \
    --pesos models/yolov3_custom_last_13122024.weights \
    --config models/yolov3_custom.cfg
```

**Detectar en video / webcam / cámara de seguridad (IP o RTSP):**

```bash
python main.py --fuente 0                                       # webcam local
python main.py --fuente video.mp4 --guardar salida.mp4          # archivo, guardando el resultado
python main.py --fuente http://192.168.0.10:8080/video          # cámara IP (ej. IP Webcam)
python main.py --fuente rtsp://usuario:pass@192.168.1.50:554/stream1  # cámara de seguridad RTSP
```

Agrega `--sin-ventana` para correrlo en un servidor sin entorno gráfico
(solo guarda el resultado, no abre ventana de OpenCV).

**Evaluar el modelo contra un dataset anotado (formato YOLO):**

```bash
python scripts/evaluar_modelo.py --dataset ruta/a/dataset
```

**Uso como librería:**

```python
import cv2
from deteccion_armas import WeaponDetector

detector = WeaponDetector(
    weights_path="models/yolov3_custom_last_13122024.weights",
    config_path="models/yolov3_custom.cfg",
    confidence_threshold=0.5,
)

image = cv2.imread("ejemplo.jpg")
annotated, detections = detector.detect_and_draw(image)
print(f"{len(detections)} arma(s) detectada(s)")
```

## Entrenamiento (Darknet)

El modelo se entrenó con [Darknet](https://github.com/AlexeyAB/darknet)
(YOLOv3, transfer learning desde pesos preentrenados en COCO, 10 000
iteraciones) sobre un dataset propio de imágenes de armas de fuego anotadas
en formato YOLO. Este repo no incluye el framework Darknet ni el dataset de
entrenamiento — solo el código de inferencia/evaluación sobre el modelo ya
entrenado. Si quieres reentrenar desde cero, cloná Darknet directamente y
usá `models/yolov3_custom.cfg` como punto de partida de la arquitectura.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Cubren el cálculo de IoU (`intersection_over_union`), que es la base tanto
del filtrado por confianza/NMS como de la evaluación contra ground truth.
El pipeline completo (`main.py` + `WeaponDetector` + pesos reales) también
se verificó manualmente contra `results/samples/`.

> **Nota sobre versiones de OpenCV**: `opencv-python 5.0.0` eliminó el
> importador de modelos Darknet (`cv2.dnn.readNet` ya no puede leer
> `.cfg`/`.weights`). Por eso `requirements.txt` fija `opencv-python<5`
> explícitamente — sin ese pin, `pip install` de hoy en adelante instala una
> versión que rompe este proyecto.


## Cita

Si usas este trabajo, por favor cita la tesis original:

```
Holguín Mori, J. K. (2025). Implementación de algoritmo de redes neuronales
convolucionales para la identificación anticipada de armas de fuego.
[Tesis de licenciatura, Universidad Nacional Mayor de San Marcos].
```

## Licencia

[MIT](LICENSE) — código de inferencia/evaluación. Los pesos entrenados y el
dataset no están cubiertos por esta licencia (ver [`models/README.md`](models/README.md)).
