# model/morphology.py
"""
Morphologie : top-hat pour E-ophtha-EX (exsudats CLAIRS).
Top-hat blanc = image − ouverture(image, SE) → isole les structures CLAIRES
plus petites que le SE. C'est l'opérateur adapté aux exsudats.
"""
import cv2
import numpy as np
from typing import Dict, Any
from model.interfaces import IMorphology


class MorphologyProcessor(IMorphology):

    def process(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        morpho_type = params.get("morpho_type", "tophat")
        if morpho_type == "none":
            return image.copy()

        radius = int(params.get("morpho_radius", 15))
        if radius < 1:
            radius = 1
        se_size = 2 * radius + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))

        iterations = int(params.get("morpho_iterations", 1))

        if morpho_type == "tophat":
            result = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se,
                                      iterations=iterations)
        elif morpho_type == "blackhat":
            result = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, se,
                                      iterations=iterations)
        else:
            result = image.copy()

        # Nettoyage léger : ouverture pour éliminer bruit poivre
        cleanup = int(params.get("cleanup_size", 1))
        if cleanup > 0:
            se_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (2 * cleanup + 1, 2 * cleanup + 1))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, se_c)

        return result