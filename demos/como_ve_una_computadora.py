#!/usr/bin/env python3
"""Demo didáctica: descompone una imagen en sus canales B/G/R y guarda los
valores de cada canal como matrices de texto — usada en la sustentación
para explicar visualmente cómo una imagen es, para la computadora, solo una
matriz de números por canal de color.

Uso:
    python demos/como_ve_una_computadora.py --imagen ruta/a/imagen.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagen", required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--tamano", type=int, default=50, help="Lado (px) al que se redimensiona antes de exportar")
    parser.add_argument("--salida", default="results", help="Carpeta donde guardar los .txt de cada canal")
    parser.add_argument("--mostrar", action="store_true", help="Mostrar la imagen original en una ventana")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.imagen)
    if image is None:
        raise SystemExit(f"No se pudo leer la imagen: {args.imagen}")

    if args.mostrar:
        cv2.imshow("imagen original", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    small = cv2.resize(image, (args.tamano, args.tamano))
    blue, green, red = cv2.split(small)

    for name, channel in (("blue", blue), ("green", green), ("red", red)):
        np.savetxt(salida / f"{name}_channel_values.txt", channel, fmt="%d")

    print(f"Valores de los 3 canales guardados en: {salida}/")


if __name__ == "__main__":
    main()
