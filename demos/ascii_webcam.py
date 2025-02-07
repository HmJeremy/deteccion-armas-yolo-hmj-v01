#!/usr/bin/env python3
"""Demo didáctica: convierte el feed de la webcam a arte ASCII en la
terminal en tiempo real. No tiene relación con la detección de armas — se
usó en la sustentación para explicar de forma visual cómo una computadora
"ve" una imagen como una matriz de números.

Uso: python demos/ascii_webcam.py
"""

from __future__ import annotations

import os
import subprocess

import cv2

ASCII_RAMP = (' ', '.', "'", ',', ':', ';', 'c', 'l',
              'x', 'o', 'k', 'X', 'd', 'O', '0', 'K', 'N')


class AsciiWebcam:
    def __init__(self, camera_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {camera_index}")

    @staticmethod
    def _row_to_ascii(row) -> tuple[str, ...]:
        return tuple(ASCII_RAMP[int(pixel / (255 / 16))] for pixel in row)[::-1]

    def _frame_to_ascii(self, gray_frame) -> tuple[tuple[str, ...], ...]:
        return tuple(self._row_to_ascii(row) for row in gray_frame)

    @staticmethod
    def _terminal_size() -> tuple[int, int]:
        size = os.get_terminal_size()
        return size.lines, size.columns

    @staticmethod
    def _clear_terminal() -> None:
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def _rescale(frame, percent: float = 50):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def run(self) -> None:
        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    break

                rows, cols = self._terminal_size()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                reduced = cv2.resize(gray, (cols, rows))

                self._clear_terminal()
                ascii_art = self._frame_to_ascii(reduced)
                print("\n".join("".join(row) for row in ascii_art), end="")

                cv2.imshow("webcam", self._rescale(cv2.flip(frame, 1)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    AsciiWebcam().run()
