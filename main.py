#!/usr/bin/env python3
"""Este es el único comando que necesitas para correr el detector: le pasas
una imagen, un video, la webcam o una cámara de seguridad IP y hace lo que
corresponda.

La fuente se interpreta sola, no hay que decirle qué tipo es:
    - un número ("0", "1", ...)                          -> webcam local
    - algo que termina en .jpg/.png/.jpeg/.bmp            -> imagen fija
    - cualquier otra cosa (.mp4, rtsp://, http://ip:8080/video, etc.)
      -> lo trata como stream de video, frame a frame

Ejemplos:
    python main.py --fuente foto.jpg
    python main.py --fuente 0
    python main.py --fuente video.mp4 --guardar salida.mp4
    python main.py --fuente rtsp://usuario:pass@192.168.1.50:554/stream1
    python main.py --fuente http://192.168.0.10:8080/video

Necesita los pesos entrenados en `models/` (no vienen en el repo por el
peso) o que le pases las rutas con --pesos/--config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from deteccion_armas import WeaponDetector  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fuente", required=True,
        help="Imagen, video, índice de webcam (0,1,...) o URL de cámara IP/RTSP",
    )
    parser.add_argument("--pesos", default="models/yolov3_custom_last_13122024.weights", help="Ruta al .weights")
    parser.add_argument("--config", default="models/yolov3_custom.cfg", help="Ruta al .cfg")
    parser.add_argument("--confianza", type=float, default=0.5, help="Umbral mínimo de confianza (0-1)")
    parser.add_argument("--nms", type=float, default=0.4, help="Umbral de IoU para non-maxima suppression")
    parser.add_argument("--guardar", default=None, help="Ruta de salida (imagen anotada, o video si la fuente es video/cámara)")
    parser.add_argument("--sin-ventana", action="store_true", help="No abrir ventana de visualización (útil en servidores sin GUI)")
    parser.add_argument("--ancho-ventana", type=int, default=900)
    parser.add_argument("--alto-ventana", type=int, default=600)
    return parser.parse_args()


def is_image_source(fuente: str) -> bool:
    return Path(fuente.split("?")[0]).suffix.lower() in IMAGE_EXTENSIONS


def resolve_video_source(fuente: str) -> int | str:
    return int(fuente) if fuente.isdigit() else fuente


def run_on_image(detector: WeaponDetector, fuente: str, args: argparse.Namespace) -> None:
    image = cv2.imread(fuente)
    if image is None:
        raise SystemExit(f"No se pudo leer la imagen: {fuente}")

    annotated, detections = detector.detect_and_draw(image)

    print(f"{len(detections)} detección(es):")
    for det in detections:
        print(f"  - {detector.classes[det.class_id]}  confianza={det.confidence:.2%}  caja={det.box}")

    salida = args.guardar or str(Path(fuente).with_stem(Path(fuente).stem + "_detectado"))
    cv2.imwrite(salida, annotated)
    print(f"Imagen anotada guardada en: {salida}")

    if not args.sin_ventana:
        cv2.imshow("Deteccion de armas", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(detector: WeaponDetector, fuente: str, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(resolve_video_source(fuente))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente de video: {fuente}")

    writer = None
    if args.guardar:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.ancho_ventana
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.alto_ventana
        writer = cv2.VideoWriter(args.guardar, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not args.sin_ventana:
        print("Presiona 'q' para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fin del video / stream, o no se pudo leer un frame.")
                break

            start = time.time()
            annotated, detections = detector.detect_and_draw(frame)
            fps_now = 1 / (time.time() - start + 1e-9)

            cv2.putText(
                annotated, f"{fps_now:.1f} FPS | {len(detections)} deteccion(es)",
                (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.sin_ventana:
                cv2.imshow(
                    "Deteccion de armas",
                    cv2.resize(annotated, (args.ancho_ventana, args.alto_ventana)),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    detector = WeaponDetector(
        args.pesos, args.config,
        confidence_threshold=args.confianza,
        nms_threshold=args.nms,
    )

    if is_image_source(args.fuente):
        run_on_image(detector, args.fuente, args)
    else:
        run_on_video(detector, args.fuente, args)


if __name__ == "__main__":
    main()
