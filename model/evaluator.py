# model/evaluator.py
"""
Évaluation : precision/recall/f1 via scikit-learn (pixel-level) + lesion-level.
Gère le cas image saine (GT vide) sans division par zéro.
"""
import numpy as np
import cv2
from typing import Dict, List
from sklearn.metrics import precision_recall_fscore_support
from model.interfaces import IEvaluator


class Evaluator(IEvaluator):

    def compute_metrics(self, pred_mask: np.ndarray,
                        gt_mask: np.ndarray) -> Dict[str, float]:
        pred = (pred_mask.flatten() > 0).astype(np.uint8)
        gt = (gt_mask.flatten() > 0).astype(np.uint8)

        if gt.sum() == 0 and pred.sum() == 0:
            return {
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "n_pred_pixels": 0, "n_gt_pixels": 0,
                "n_tp": 0, "n_fp": 0, "n_fn": 0,
            }

        precision, recall, f1, _ = precision_recall_fscore_support(
            gt, pred, average="binary", zero_division=0)

        tp = int(np.sum((pred == 1) & (gt == 1)))
        fp = int(np.sum((pred == 1) & (gt == 0)))
        fn = int(np.sum((pred == 0) & (gt == 1)))

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_pred_pixels": int(pred.sum()),
            "n_gt_pixels": int(gt.sum()),
            "n_tp": tp, "n_fp": fp, "n_fn": fn,
        }

    def compute_lesion_level(self, pred_components: List[dict],
                             gt_mask: np.ndarray,
                             iou_threshold: float = 0.3) -> Dict[str, float]:
        """
        Évaluation objet : une prédiction est TP si son centroïde tombe dans
        un objet GT. Simplification acceptable pour un projet académique.
        """
        num_gt, _, gt_stats, _ = cv2.connectedComponentsWithStats(
            gt_mask, connectivity=8)
        n_gt_objects = max(0, num_gt - 1)

        tp = 0
        for comp in pred_components:
            cx, cy = comp["centroid"]
            xi, yi = int(cx), int(cy)
            if 0 <= yi < gt_mask.shape[0] and 0 <= xi < gt_mask.shape[1]:
                if gt_mask[yi, xi] > 0:
                    tp += 1

        fp = len(pred_components) - tp
        fn = max(0, n_gt_objects - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "precision_obj": precision,
            "recall_obj": recall,
            "f1_obj": f1,
            "n_pred_obj": len(pred_components),
            "n_gt_obj": n_gt_objects,
        }