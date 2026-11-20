# model/segmenter.py
"""
Segmentation : seuillage + composantes connexes + filtrage (aire + circularité).
Pour EX : aire min plus grande (20), circularité plus permissive (0.2).
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from model.interfaces import ISegmenter


class Segmenter(ISegmenter):

    def process(self, image: np.ndarray, params: Dict[str, Any],
                roi_mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        # 1. Seuillage
        method = params.get("threshold_method", "otsu")

        if method == "otsu":
            thresh_val, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            offset = int(params.get("otsu_offset", 0))
            if offset != 0:
                thresh_val = float(np.clip(thresh_val + offset, 0, 255))
                _, binary = cv2.threshold(image, thresh_val, 255,
                                          cv2.THRESH_BINARY)
        elif method == "fixed":
            val = int(params.get("fixed_threshold", 30))
            _, binary = cv2.threshold(image, val, 255, cv2.THRESH_BINARY)
        elif method == "adaptive":
            block = int(params.get("adaptive_block", 15))
            if block % 2 == 0:
                block += 1
            if block < 3:
                block = 3
            C = float(params.get("adaptive_C", 2))
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, C)
        else:
            binary = np.zeros_like(image, dtype=np.uint8)

        # 2. Restriction au disque rétinien
        binary = cv2.bitwise_and(binary, roi_mask)

        # 3. Post-traitement morphologique
        open_k = int(params.get("open_post", 1))
        close_k = int(params.get("close_post", 1))
        if open_k > 0:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * open_k + 1, 2 * open_k + 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
        if close_k > 0:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * close_k + 1, 2 * close_k + 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)

        # 4. Composantes connexes
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        # 5. Filtrage
        min_area = int(params.get("min_area", 20))
        max_area = int(params.get("max_area", 5000))
        min_circ = float(params.get("min_circularity", 0.2))

        components: List[dict] = []
        filtered_mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
            if circularity < min_circ:
                continue

            filtered_mask[labels == i] = 255
            components.append({
                "area": area,
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                "circularity": float(circularity),
            })

        return filtered_mask, components