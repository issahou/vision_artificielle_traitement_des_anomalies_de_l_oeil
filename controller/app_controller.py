# controller/app_controller.py
"""
Médiateur entre View et Model. Toute la coordination ici.
"""
import os
import json
import csv
import cv2
import numpy as np
from typing import Optional, Dict, Any
from model.pipeline import Pipeline
from model.data_loader import DataLoader
from view.main_window import MainWindow


class AppController:

    def __init__(self, dataset_root: str):
        self.view = MainWindow()
        self.pipeline = Pipeline()
        self.data_loader = DataLoader(dataset_root)

        self.current_image: Optional[np.ndarray] = None
        self.current_gt: Optional[np.ndarray] = None
        self.current_filename: str = ""
        self.last_results: Optional[Dict[str, Any]] = None

        # Branchement des callbacks
        self.view.on_load_image = self.load_image
        self.view.on_load_gt = self.load_gt
        self.view.on_params_changed = self.run_pipeline
        self.view.on_save_results = self.save_results
        self.view.on_sweep_threshold = self.sweep_thresholds

    # ---------- Chargements ----------
    def load_image(self):
        try:
            path = self.view.ask_image_path()
            if not path:
                return
            self.current_image = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.current_image is None:
                raise ValueError("Image illisible")
            self.current_filename = os.path.basename(path)

            # Tentative de chargement automatique du GT (même nom dans Annotation_EX/)
            base = os.path.splitext(self.current_filename)[0]
            possible_gt = os.path.join(self.data_loader.mask_dir, base + ".png")
            if os.path.exists(possible_gt):
                self.current_gt = self.data_loader.load_gt_mask(self.current_filename)
                self.view.set_info(f"Image + GT auto : {self.current_filename}")
            else:
                self.current_gt = None
                self.view.set_info(f"Image : {self.current_filename} (pas de GT)")

            self.run_pipeline()
        except Exception as e:
            self.view.show_error(str(e))

    def load_gt(self):
        try:
            if self.current_image is None:
                self.view.show_error("Charge d'abord une image.")
                return
            path = self.view.ask_gt_path()
            if not path:
                return
            gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise ValueError("Masque GT illisible")
            self.current_gt = (gt > 0).astype(np.uint8) * 255
            self.view.set_info(f"GT chargé : {os.path.basename(path)}")
            self.run_pipeline()
        except Exception as e:
            self.view.show_error(str(e))

    # ---------- Pipeline ----------
    def run_pipeline(self):
        if self.current_image is None:
            return
        try:
            params = self.view.get_params()
            results = self.pipeline.run(self.current_image, params, self.current_gt)
            self.last_results = results
            self.view.display_results(results)
        except Exception as e:
            self.view.show_error(f"Erreur pipeline : {e}")

    # ---------- Export ----------
    def save_results(self):
        if self.last_results is None:
            self.view.show_error("Aucun résultat à sauvegarder.")
            return
        try:
            out = "output"
            os.makedirs(out, exist_ok=True)
            base = os.path.splitext(self.current_filename)[0] or "result"

            for key in ["preprocessed", "morphology", "pred_mask", "overlay"]:
                img = self.last_results.get(key)
                if img is not None:
                    cv2.imwrite(os.path.join(out, f"{base}_{key}.png"), img)

            with open(os.path.join(out, f"{base}_params.json"), "w") as f:
                json.dump(self.view.get_params(), f, indent=2)

            self.view.set_info(f"Sauvegardé dans {out}/")
        except Exception as e:
            self.view.show_error(str(e))

    # ---------- Balayage seuils (courbe PR) ----------
    def sweep_thresholds(self):
        if self.current_image is None or self.current_gt is None:
            self.view.show_error("Image + GT requis pour le balayage.")
            return
        try:
            base_params = self.view.get_params()
            offsets = list(range(-60, 61, 5))
            rows = []
            for off in offsets:
                p = dict(base_params)
                p["otsu_offset"] = off
                res = self.pipeline.run(self.current_image, p, self.current_gt)
                m = res.get("metrics", {})
                rows.append({
                    "offset": off,
                    "precision": m.get("precision", 0.0),
                    "recall": m.get("recall", 0.0),
                    "f1": m.get("f1", 0.0),
                })

            out = "output"
            os.makedirs(out, exist_ok=True)
            base = os.path.splitext(self.current_filename)[0] or "result"
            csv_path = os.path.join(out, f"{base}_pr_sweep.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["offset", "precision", "recall", "f1"])
                w.writeheader()
                w.writerows(rows)

            best = max(rows, key=lambda r: r["f1"])
            self.view.show_info(
                f"Balayage terminé.\n"
                f"Meilleur F1 = {best['f1']:.3f} (offset={best['offset']})\n"
                f"CSV : {csv_path}"
            )
        except Exception as e:
            self.view.show_error(str(e))

    def run(self):
        self.view.mainloop()