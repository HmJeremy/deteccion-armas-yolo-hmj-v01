# Detector de Armas de Fuego en Tiempo Real

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](requirements.txt)
[![OpenCV DNN](https://img.shields.io/badge/inference-OpenCV%20DNN-green)](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)

Detección anticipada de armas de fuego en flujo de video mediante **YOLOv3** optimizado para producción.

* **Entrenamiento:** Modelo entrenado en Darknet utilizando un dataset propio.
* **Inferencia / Producción:** Ejecución optimizada mediante el módulo `cv2.dnn` de OpenCV. No requiere dependencias de Darknet para el despliegue.


> *Implementación de algoritmo de redes neuronales convolucionales para la identificación anticipada de armas de fuego (UNMSM - CC, 2025).*


<p align="center">
  <img src="results/samples/resultado_05.jpg" height="250" alt="Detección de arma parcialmente oculta">
  <img src="results/samples/resultado_12.jpg" height="250" alt="Detección de réplica de arma">
  <img src="results/samples/resultado_20.jpg" height="250" alt="Detección de arma">
</p>


## Flujo

<div align="center">

| Pipeline de Procesamiento (Inferencia) |
| :---: |
| **Imagen / Frame de Video** |
| ↓ |
| **Blob 416×416** <br> *Se adapta la imagen al tamaño y formato exacto que el algoritmo necesita.* |
| ↓ |
| **Forward Pass YOLOv3** <br> *La red neuronal analiza la imagen adaptada buscando armas.* |
| ↓ |
| **Filtro de Confianza** <br> *Se descartan las detecciones que no superan el umbral de confianza.* |
| ↓ |
| **Non-Maxima Suppression (NMS)** <br> *Si hay varios cuadros sobre una misma arma, se borran los repetidos y queda el mejor detectado.* |
| ↓ |
| **Cajas Finales (Detección)** <br> *El resultado final: un recuadro limpio dibujado sobre el arma detectada.* |

</div>


## Estructura del Proyecto

```text
├── src/
│   └── deteccion_armas/
│       ├── detector.py   # Clase WeaponDetector: carga de modelo YOLOv3 e inferencia (detect/draw)
│       └── evaluar.py    # Métricas de rendimiento: cálculo de IoU, matriz de confusión y Precision/Recall/F1
├── scripts/
│   └── evaluar_modelo.py # Pipeline de evaluación automatizada sobre datasets anotados
├── demos/                # Módulos educativos: visualización BGR y renderizado ASCII para la sustentación
└── main.py               # CLI (Punto de entrada): soporta fuentes de entrada locales (imagen/video), Webcam e IP/RTSP
```


## Instalación

```bash
git clone deteccion-armas-yolo-hmj-v01
cd deteccion-armas-yolo-hmj-v01
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Después necesitas los pesos entrenados. Solicitar a 39601508+HmJeremy@users.noreply.github.com

## Uso

`main.py` es el único comando que necesitas: detecta solo si le diste una
imagen o un video/stream en vivo, y aplica el flujo que corresponde.

**Detectar en una imagen:**

```bash
python main.py --fuente ejemplo.jpg \
    --pesos models/yolov3_weapons.weights \
    --config models/yolov3_weapons.cfg
```

**Detectar en video / webcam / cámara de seguridad (IP o RTSP):**

```bash
python main.py --fuente 0                                       # webcam local
python main.py --fuente video.mp4 --guardar salida.mp4          # archivo, guardando el resultado
python main.py --fuente http://192.168.0.10:8080/video          # cámara IP (ej. IP Webcam)
python main.py --fuente rtsp://usuario:pass@192.168.1.50:554/stream1  # cámara de seguridad RTSP
```

Para correrlo en un servidor sin entorno gráfico, agréga `--sin-ventana`
(solo guarda el resultado, no intenta abrir una ventana de OpenCV).

**Evaluar el modelo contra un dataset con anotaciones (formato YOLO):**

```bash
python scripts/evaluar_modelo.py --dataset cambiar/ruta/dataset
```

**Usándolo como librería:**

```python
import cv2
from deteccion_armas import WeaponDetector

detector = WeaponDetector(
    weights_path="models/yolov3_weapons.weights",
    config_path="models/yolov3_weapons.cfg",
    confidence_threshold=0.5,
)

image = cv2.imread("ejemplo.jpg")
annotated, detections = detector.detect_and_draw(image)
print(f"{len(detections)} arma(s) detectada(s)")
```

## Entrenamiento (Darknet)

El modelo fue desarrollado mediante **Transfer Learning** sobre la arquitectura **YOLOv3**, utilizando como base los pesos preentrenados de MS COCO y completando 10,000 iteraciones de *fine-tuning* sobre un dataset personalizado y anotado en formato YOLO.

> **Nota:** Este repositorio contiene exclusivamente el entorno de inferencia y evaluación. No incluye el framework Darknet ni el dataset crudo de entrenamiento.

### Reentrenamiento (Fine-tuning desde cero)
Si requieres compilar o reentrenar el modelo:
1. Clona de forma independiente el repositorio oficial de [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).
2. Utiliza el archivo de configuración de arquitectura provisto en este repositorio: `models/yolov3_weapons.cfg`.


## Testing

### Pruebas Unitarias
El proyecto implementa pruebas unitarias para validar la consistencia del cálculo matemático de **Intersection over Union (IoU)**, componente crítico tanto para la supresión de no máximos (NMS) como para la evaluación de métricas de rendimiento frente al *ground truth*.

Ejecución del set de pruebas:
```bash
pip install pytest
pytest tests/ -v
```

> **Nota:** Las pruebas sobre el pipeline completo de inferencia (`main.py` + `WeaponDetector` + artefactos de pesos) fueron validadas mediante pruebas de regresión visual utilizando las muestras ubicadas en `results/samples/`.

### Dependencia Crítica: Versión de OpenCV (Breaking Change)
El archivo `requirements.txt` restringe estrictamente la versión de OpenCV (`opencv-python < 5.0.0`). 

A partir de la versión `5.0.0`, el backend de OpenCV **removió el soporte nativo para Darknet**, por lo que el método `cv2.dnn.readNet` ya no es compatible con archivos `.cfg` y `.weights`. Omitir esta restricción romperá el entorno de inferencia del proyecto.


## Cita

Si usas este trabajo, por favor cita la tesis original:

```
Holguín Mori, J. K. (2025). Implementación de algoritmo de redes neuronales
convolucionales para la identificación anticipada de armas de fuego.
[Tesis de licenciatura, Universidad Nacional Mayor de San Marcos].
```

## Licencia

[MIT](LICENSE) para el código de inferencia/evaluación. Los pesos
entrenados y el dataset no están cubiertos por esta licencia.
