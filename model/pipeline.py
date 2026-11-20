# model/pipeline.py
"""
Orchestrateur du pipeline. Dépend d'interfaces (SOLID-D).
Retourne TOUTES les images intermédiaires + métriques.

CORRECTIFS v1.1 :
- Érosion du ROI pour supprimer les artéfacts d'anneau de bord
  (le fond flou 71px crée un halo brillant à la frontière du disque
   qui génère des faux positifs massifs sur E-ophtha-EX).
- Alignement du GT corrigé : on applique la MÊME bounding box que celle
  du recadrage (avant : simple resize → lésions décalées → métriques fausses).
- Gestion cas limites : ROI vide, GT None/vide, shapes différentes.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from model.interfaces import IPreprocessor, IMorphology, ISegmenter, IEvaluator
from model.preprocessor import Preprocessor
from model.morphology import MorphologyProcessor
from model.segmenter import Segmenter
from model.evaluator import Evaluator


class Pipeline:

    def __init__(self,
                 preprocessor: Optional[IPreprocessor] = None,
                 morphology: Optional[IMorphology] = None,
                 segmenter: Optional[ISegmenter] = None,
                 evaluator: Optional[IEvaluator] = None):
        self.preprocessor = preprocessor or Preprocessor()
        self.morphology = morphology or MorphologyProcessor()
        self.segmenter = segmenter or Segmenter()
        self.evaluator = evaluator or Evaluator()

    # ------------------------------------------------------------------ #
    # Étape 0 : Détection du disque rétinien (ROI)
    # ------------------------------------------------------------------ #
    def detect_roi_mask(self, image: np.ndarray,
                        params: Dict[str, Any]) -> np.ndarray:
        """
        Détecte le disque rétinien (zone utile) et retourne un masque binaire.
        Étapes :
          1. Gris + Otsu (robuste à l'illumination)
          2. Fermeture morphologique (lisser le contour)
          3. Plus grande composante connexe
          4. Remplissage des trous
          5. Érosion de 5 px pour exclure les artéfacts de bord
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Seuillage Otsu
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Fermeture (lisser contour)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 3. Plus grande composante
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        if num <= 1:
            # Cas dégénéré : aucun objet détecté → tout est ROI
            return np.ones_like(gray, dtype=np.uint8) * 255

        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest_idx).astype(np.uint8) * 255

        # 4. Remplir les trous
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)

        # 5. Érosion pour exclure l'anneau d'artéfact de bord
        #    (causé par la correction d'illumination gaussienne de grand rayon)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        filled = cv2.erode(filled, erode_kernel)

        return filled

    # ------------------------------------------------------------------ #
    # Pipeline principal
    # ------------------------------------------------------------------ #
    def run(self, image: np.ndarray, params: Dict[str, Any],
            gt_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Exécute tout le pipeline.
        Retourne un dict contenant :
          - toutes les images intermédiaires (pour l'affichage multi-vues)
          - les métriques (si gt_mask fourni)
          - la bounding box de recadrage (pour aligner le GT)
        """
        result: Dict[str, Any] = {}
        result["original"] = image.copy()

        # ------------------------------------------------------------ #
        # 0. ROI
        # ------------------------------------------------------------ #
        roi_mask = self.detect_roi_mask(image, params)
        result["roi_mask"] = roi_mask

        # ------------------------------------------------------------ #
        # 0bis. Recadrage optionnel — on garde la bbox pour aligner le GT
        # ------------------------------------------------------------ #
        work_img = image
        work_roi = roi_mask
        crop_box: Optional[Tuple[int, int, int, int]] = None

        if params.get("crop", True):
            coords = cv2.findNonZero(roi_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                if w > 0 and h > 0:
                    work_img = image[y:y + h, x:x + w]
                    work_roi = roi_mask[y:y + h, x:x + w]
                    crop_box = (x, y, w, h)
                    result["cropped"] = work_img.copy()
                    result["crop_box"] = crop_box

        # ------------------------------------------------------------ #
        # 1. Prétraitement
        # ------------------------------------------------------------ #
        preprocessed = self.preprocessor.process(work_img, params)
        result["preprocessed"] = preprocessed

        # ------------------------------------------------------------ #
        # 2. Morphologie (top-hat pour EX)
        # ------------------------------------------------------------ #
        morpho = self.morphology.process(preprocessed, params)
        result["morphology"] = morpho

        # ------------------------------------------------------------ #
        # 3. Segmentation
        # ------------------------------------------------------------ #
        pred_mask, components = self.segmenter.process(morpho, params, work_roi)
        result["pred_mask"] = pred_mask
        result["components"] = components

        # ------------------------------------------------------------ #
        # 4. Overlay (lésions détectées en rouge BGR)
        # ------------------------------------------------------------ #
        overlay = work_img.copy()
        overlay[pred_mask > 0] = (0, 0, 255)
        result["overlay"] = overlay

        # ------------------------------------------------------------ #
        # 5. Métriques (si GT disponible)
        # ------------------------------------------------------------ #
        if gt_mask is not None:
            gt_aligned = self._align_gt_to_pred(gt_mask, crop_box, pred_mask)

            # Sécurité : si après alignement les shapes diffèrent encore
            if gt_aligned.shape != pred_mask.shape:
                gt_aligned = cv2.resize(
                    gt_aligned,
                    (pred_mask.shape[1], pred_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST)

            # Binarisation (255 = lésion)
            gt_aligned = (gt_aligned > 0).astype(np.uint8) * 255

            pixel_m = self.evaluator.compute_metrics(pred_mask, gt_aligned)
            lesion_m = self.evaluator.compute_lesion_level(
                components, gt_aligned)
            result["metrics"] = {**pixel_m, **lesion_m}
            result["gt_mask"] = gt_aligned

        return result

    # ------------------------------------------------------------------ #
    # Helper : aligner le GT sur la même zone que l'image de travail
    # ------------------------------------------------------------------ #
    @staticmethod
    def _align_gt_to_pred(gt_mask: np.ndarray,
                          crop_box: Optional[Tuple[int, int, int, int]],
                          pred_mask: np.ndarray) -> np.ndarray:
        """
        Aligne le GT sur la même zone que l'image après recadrage.

        PROBLÈME RÉSOLU :
          Avant : on faisait juste cv2.resize(gt, pred.shape) → les lésions
          du GT étaient DÉCALÉES si le recadrage avait décalé l'image.
          Ici : on applique la MÊME bounding box (x, y, w, h) au GT, donc
          les lésions restent exactement au bon endroit.

        Cas gérés :
          - crop_box = None (pas de recadrage) : le GT est déjà aligné
          - crop_box hors limites du GT : on clampe
          - gt plus petit que crop_box : resize de secours (rare)
        """
        if crop_box is None:
            return gt_mask

        x, y, w, h = crop_box
        gh, gw = gt_mask.shape[:2]

        # Clamper la bbox dans les limites du GT
        x0 = max(0, min(x, gw - 1))
        y0 = max(0, min(y, gh - 1))
        x1 = max(x0 + 1, min(x + w, gw))
        y1 = max(y0 + 1, min(y + h, gh))

        cropped = gt_mask[y0:y1, x0:x1]

        # Cas rare : dimensions du crop_box dépassent le GT → resize
        if cropped.shape[0] != h or cropped.shape[1] != w:
            cropped = cv2.resize(cropped, (w, h),
                                 interpolation=cv2.INTER_NEAREST)

        return cropped