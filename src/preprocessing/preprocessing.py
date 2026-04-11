from __future__ import annotations

import cv2
import numpy as np


class FramePreprocessor:
    def __init__(self) -> None:
        pass

    def get_clean_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Resize (important pour stabilité + vitesse)
        frame = cv2.resize(frame, (640, 480))

        # Réduction du bruit (eau)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Conversion en gris
        gray_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        return blurred, gray_frame