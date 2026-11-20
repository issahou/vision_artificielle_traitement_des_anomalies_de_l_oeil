# model/preprocessor.py
"""
Prétraitement : canal vert → correction illumination → CLAHE → flou → unsharp.
Ordre critique : illumination AVANT CLAHE, flou APRÈS CLAHE, unsharp EN DERNIER.
"""
import cv2
import numpy as np
from typing import Dict, Any
from model.interfaces import IPreprocessor


class Preprocessor(IPreprocessor):

    def process(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        # 1. Extraction canal
        channel = params.get("channel", "green")
        if channel == "green":
            gray = image[:, :, 1].copy()
        elif channel == "red":
            gray = image[:, :, 2].copy()
        elif channel == "blue":
            gray = image[:, :, 0].copy()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Correction d'illumination (soustraction du fond)
        illum_size = int(params.get("illum_blur_size", 71))
        if illum_size % 2 == 0:
            illum_size += 1
        background = cv2.GaussianBlur(gray, (illum_size, illum_size), 0)
        corrected = cv2.subtract(gray, background)
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        # 3. CLAHE
        if params.get("clahe_enabled", True):
            clip = float(params.get("clahe_clip", 3.0))
            tile = int(params.get("clahe_tile", 8))
            if tile < 2:
                tile = 2
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            corrected = clahe.apply(corrected)

        # 4. Flou (réduction bruit)
        blur_type = params.get("blur_type", "median")
        blur_size = int(params.get("blur_size", 3))
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_type == "median" and blur_size > 1:
            corrected = cv2.medianBlur(corrected, blur_size)
        elif blur_type == "gaussian" and blur_size > 1:
            corrected = cv2.GaussianBlur(corrected, (blur_size, blur_size), 0)

        # 5. Unsharp masking (rehaussement netteté)
        strength = float(params.get("unsharp_strength", 0.5))
        if strength > 0:
            blurred = cv2.GaussianBlur(corrected, (0, 0), 3)
            corrected = cv2.addWeighted(corrected, 1.0 + strength,
                                        blurred, -strength, 0)
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        return corrected