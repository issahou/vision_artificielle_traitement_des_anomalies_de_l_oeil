# model/data_loader.py
"""
Chargement des images et masques E-ophtha-EX.
Structure : EX/ (images), Annotation_EX/ (masques), healthy/ (négatifs).
"""
import cv2
import numpy as np
import os
from typing import Optional, List


class DataLoader:
    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.img_dir = os.path.join(dataset_root, "EX")
        self.mask_dir = os.path.join(dataset_root, "Annotation_EX")
        self.healthy_dir = os.path.join(dataset_root, "healthy")

    def load_image(self, filename: str) -> np.ndarray:
        path = os.path.join(self.img_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {path}")
        return img

    def load_gt_mask(self, filename: str) -> Optional[np.ndarray]:
        base = os.path.splitext(filename)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")
        if not os.path.exists(mask_path):
            return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        return (mask > 0).astype(np.uint8) * 255

    def list_positive_images(self) -> List[str]:
        if not os.path.exists(self.img_dir):
            return []
        return sorted([f for f in os.listdir(self.img_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def list_healthy_images(self) -> List[str]:
        if not os.path.exists(self.healthy_dir):
            return []
        return sorted([f for f in os.listdir(self.healthy_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def load_healthy_image(self, filename: str) -> np.ndarray:
        path = os.path.join(self.healthy_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {path}")
        return img