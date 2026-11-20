# model/interfaces.py
"""
Interfaces abstraites (SOLID-I, SOLID-D).
Chaque étape du pipeline implémente une interface minimale et dédiée.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, List


class IPreprocessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Retourne l'image prétraitée (gris, contraste amélioré)."""
        pass


class IMorphology(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Retourne l'image après opération morphologique (top-hat pour EX)."""
        pass


class ISegmenter(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, params: Dict[str, Any],
                roi_mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Retourne (masque_binaire, liste_composantes).
        Chaque composante = dict avec area, centroid, circularity.
        """
        pass


class IEvaluator(ABC):
    @abstractmethod
    def compute_metrics(self, pred_mask: np.ndarray,
                        gt_mask: np.ndarray) -> Dict[str, float]:
        """Retourne un dict de métriques pixel-level (precision, recall, f1)."""
        pass

    @abstractmethod
    def compute_lesion_level(self, pred_components: List[dict],
                             gt_mask: np.ndarray,
                             iou_threshold: float = 0.3) -> Dict[str, float]:
        """Retourne les métriques objet-level."""
        pass