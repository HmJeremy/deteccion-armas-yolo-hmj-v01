# Detección Anticipada de Armas de Fuego con YOLO

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](requirements.txt)
[![OpenCV DNN](https://img.shields.io/badge/inference-OpenCV%20DNN-green)](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)

Este es el código de mi tesis de pregrado en la UNMSM: un detector de armas
de fuego para video de cámaras de vigilancia, funcionando en tiempo real.
Entrené una YOLOv3 con **Darknet** usando un dataset propio, y para correrla
en "producción" (o sea, en mi laptop y en las pruebas de la sustentación) uso
el módulo `cv2.dnn` de OpenCV — no hace falta instalar Darknet para usar el
detector, solo si quieres reentrenar el modelo desde cero.

El nombre completo del proyecto, para quien quiera citarlo, es
**"Implementación de algoritmo de redes neuronales convolucionales para la
identificación anticipada de armas de fuego"** (UNMSM - CC, 2025).

<p align="center">
  <img src="results/samples/resultado_05.jpg" width="410" alt="Detección de arma parcialmente oculta">
  <img src="results/samples/resultado_20.jpg" width="410" alt="Detección de réplica de arma">
</p>


## Cómo funciona

En corto, el flujo es este:

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

Y así está organizado el repo:

- **`src/deteccion_armas/detector.py`** — acá vive `WeaponDetector`, la
  clase que carga el modelo (una sola vez) y expone `detect()` para obtener
  las cajas y `draw()` para dibujarlas encima de la imagen.
- **`src/deteccion_armas/evaluar.py`** — el cálculo de IoU, matriz de
  confusión y precisión/recall/F1 contra un dataset anotado en formato YOLO.
- **`main.py`** — el punto de entrada, todo por línea de comandos: le pasas
  una imagen, un video, tu webcam o el link de una cámara IP/RTSP y hace lo
  que corresponda con el mismo comando (ver [Uso](#uso) más abajo).
- **`scripts/evaluar_modelo.py`** — corre la evaluación completa contra un
  dataset anotado (precisión, recall, F1, IoU por imagen).
- **`demos/`** — dos scriptcitos que usé en la sustentación para explicar de
  forma más visual cómo "ve" una imagen una computadora (arte ASCII de la
  webcam en vivo, y los canales BGR por separado). No tienen nada que ver
  con la detección en sí, son solo material didáctico.

## Instalación

```bash
git clone <url-de-este-repo>
cd deteccion-armas-yolo-hmj-v01
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Después necesitas los pesos entrenados. No vienen incluidos en el repo
porque pesan como 235 MB, y GitHub no deja subir archivos de más de 100 MB —
mira [`models/README.md`](models/README.md) para saber de dónde bajarlos.

## Uso

`main.py` es el único comando que necesitas: detecta solo si le diste una
imagen o un video/stream en vivo, y aplica el flujo que corresponde.

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

Si lo vas a correr en un servidor sin entorno gráfico, agrégale
`--sin-ventana` (solo guarda el resultado, no intenta abrir una ventana de
OpenCV, que ahí sí explota).

**Evaluar el modelo contra un dataset anotado (formato YOLO):**

```bash
python scripts/evaluar_modelo.py --dataset ruta/a/dataset
```

**Usándolo como librería:**

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

El modelo lo entrené con [Darknet](https://github.com/AlexeyAB/darknet)
(YOLOv3, transfer learning desde pesos preentrenados en COCO, 10 000
iteraciones) sobre un dataset propio de fotos de armas de fuego anotadas en
formato YOLO. Este repo no trae ni el framework Darknet ni el dataset de
entrenamiento, solo el código para correr inferencia/evaluación sobre el
modelo ya entrenado. Si quieres reentrenar desde cero, clona Darknet aparte
y usa `models/yolov3_custom.cfg` de este repo como punto de partida de la
arquitectura.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Los tests cubren el cálculo de IoU (`intersection_over_union`), que es la
base tanto del filtrado por confianza/NMS como de la evaluación contra el
ground truth. El pipeline completo (`main.py` + `WeaponDetector` + pesos
reales) lo probé a mano contra las imágenes de `results/samples/`, no hay
un test automatizado para eso todavía.

> **Ojo con las versiones de OpenCV**: `opencv-python 5.0.0` le quitaron el
> importador de modelos Darknet (`cv2.dnn.readNet` ya no lee
> `.cfg`/`.weights`). Por eso `requirements.txt` fija `opencv-python<5` a
> propósito — sin ese pin, un `pip install` de ahora en adelante te va a
> instalar una versión que rompe todo el proyecto.

## Cita

Si usas este trabajo, por favor cita la tesis original:

```
Holguín Mori, J. K. (2025). Implementación de algoritmo de redes neuronales
convolucionales para la identificación anticipada de armas de fuego.
[Tesis de licenciatura, Universidad Nacional Mayor de San Marcos].
```

## Licencia

[MIT](LICENSE) para el código de inferencia/evaluación. Los pesos
entrenados y el dataset no están cubiertos por esta licencia (ver
[`models/README.md`](models/README.md)).
