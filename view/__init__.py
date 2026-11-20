# test_synthetic.py
"""
Test rapide SANS dataset : vérifie que le pipeline fonctionne sur une image
synthétique contenant des cercles CLAIRS (simule des exsudats).
À lancer AVANT même de télécharger le dataset.
"""
import numpy as np
import cv2
from model.pipeline import Pipeline
from config.default_params import DEFAULT_PARAMS


def make_synthetic():
    """Crée une image 400x400 avec 3 cercles clairs sur fond gris."""
    img = np.full((400, 400, 3), 100, dtype=np.uint8)
    cv2.circle(img, (100, 100), 12, (255, 255, 255), -1)
    cv2.circle(img, (250, 200), 18, (250, 250, 250), -1)
    cv2.circle(img, (300, 300), 10, (255, 255, 255), -1)
    # Léger bruit
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_pipeline():
    img = make_synthetic()
    pipeline = Pipeline()
    params = dict(DEFAULT_PARAMS)
    params["morpho_radius"] = 20
    params["min_area"] = 15
    params["max_area"] = 5000
    params["crop"] = False

    res = pipeline.run(img, params, None)
    n = len(res["components"])
    print(f"Objets détectés : {n} (attendu : ~3)")

    # Vérifier que les 3 zones sont bien couvertes
    pred = res["pred_mask"]
    zones = [(100, 100), (250, 200), (300, 300)]
    covered = sum(1 for (x, y) in zones if pred[y, x] > 0)
    print(f"Zones couvertes : {covered}/3")

    assert n >= 2, f"❌ Attendu au moins 2 objets, obtenu {n}"
    assert covered >= 2, f"❌ Attendu 2 zones couvertes, obtenu {covered}"
    print("✅ Test pipeline OK — tu peux lancer l'application")


if __name__ == "__main__":
    test_pipeline()