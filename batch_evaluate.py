# batch_evaluate.py
"""
Évaluation quantitative sur tout le dataset E-ophtha-EX (hors interface).
Sauvegarde un CSV + affiche les moyennes.
À lancer quand les paramètres par défaut sont calibrés via l'interface.
"""
import os
import csv
import cv2
import numpy as np
from model.pipeline import Pipeline
from model.data_loader import DataLoader
from config.default_params import DEFAULT_PARAMS


def evaluate(dataset_root: str, out_csv: str = "output/eval_global.csv"):
    loader = DataLoader(dataset_root)
    pipeline = Pipeline()
    params = dict(DEFAULT_PARAMS)

    positives = loader.list_positive_images()
    healthy = loader.list_healthy_images()
    rows = []

    print(f"=== Évaluation sur {len(positives)} images positives + "
          f"{len(healthy)} images saines ===\n")

    for fname in positives:
        try:
            img = loader.load_image(fname)
            gt = loader.load_gt_mask(fname)
            if gt is None:
                continue
            res = pipeline.run(img, params, gt)
            m = res.get("metrics", {})
            rows.append({
                "filename": fname, "type": "EX",
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1", 0.0),
                "f1_obj": m.get("f1_obj", 0.0),
                "n_pred_obj": m.get("n_pred_obj", 0),
                "n_gt_obj": m.get("n_gt_obj", 0),
            })
            print(f"✅ {fname} : F1={m.get('f1', 0):.3f}")
        except Exception as e:
            print(f"❌ {fname} : {e}")

    for fname in healthy:
        try:
            img = loader.load_healthy_image(fname)
            res = pipeline.run(img, params, None)
            n_pred = len(res.get("components", []))
            rows.append({
                "filename": fname, "type": "healthy",
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "f1_obj": 0.0, "n_pred_obj": n_pred, "n_gt_obj": 0,
            })
            print(f"⚪ {fname} (saine) : {n_pred} faux positifs")
        except Exception as e:
            print(f"❌ {fname} : {e}")

    ex_rows = [r for r in rows if r["type"] == "EX"]
    if ex_rows:
        print(f"\n=== MOYENNES ({len(ex_rows)} images EX) ===")
        print(f"Précision (px) : {np.mean([r['precision'] for r in ex_rows]):.3f}")
        print(f"Rappel (px)    : {np.mean([r['recall'] for r in ex_rows]):.3f}")
        print(f"F1 (px)        : {np.mean([r['f1'] for r in ex_rows]):.3f}")
        print(f"F1 (objets)    : {np.mean([r['f1_obj'] for r in ex_rows]):.3f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV : {out_csv}")


if __name__ == "__main__":
    evaluate("data/e_ophtha_EX")