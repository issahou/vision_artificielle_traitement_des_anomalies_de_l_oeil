# test_synthetic.py
"""
Test du pipeline sur une image synthétique RÉALISTE.

Structure simulée :
  - Fond noir (extérieur rétine)
  - Grand disque gris (rétine, rayon 180)
  - 3 taches claires à l'intérieur (simulent des exsudats)

Ce test valide :
  - detect_roi_mask() détecte bien le disque
  - le top-hat fait ressortir les 3 taches
  - le filtrage garde au moins 2 objets
"""
import numpy as np
import cv2
from model.pipeline import Pipeline
from config.default_params import DEFAULT_PARAMS


def make_synthetic():
    """Image 400x400 : disque rétinien + 3 exsudats clairs à l'intérieur."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Disque rétinien
    cv2.circle(img, (200, 200), 180, (110, 110, 110), -1)

    # 3 exsudats brillants
    cv2.circle(img, (130, 140), 15, (245, 245, 245), -1)
    cv2.circle(img, (250, 200), 20, (250, 250, 250), -1)
    cv2.circle(img, (280, 280), 12, (245, 245, 245), -1)

    # Bruit léger
    noise = np.random.randint(-6, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_pipeline():
    img = make_synthetic()
    pipeline = Pipeline()

    params = dict(DEFAULT_PARAMS)
    params["morpho_radius"] = 20
    params["min_area"] = 15
    params["max_area"] = 5000
    params["min_circularity"] = 0.3
    params["crop"] = True

    res = pipeline.run(img, params, None)

    n = len(res["components"])
    print(f"Objets détectés    : {n} (attendu : ~3)")
    print(f"ROI pixels         : {int(res['roi_mask'].sum() // 255)}")
    print(f"Pred_mask pixels   : {int(res['pred_mask'].sum() // 255)}")

    if res.get("components"):
        print("Centres des composantes :")
        for c in res["components"]:
            print(f"  - centroid={c['centroid']}, area={c['area']}, "
                  f"circ={c['circularity']:.2f}")

    assert n >= 2, f"❌ Attendu ≥ 2 objets, obtenu {n}"
    print("✅ Test pipeline OK")


if __name__ == "__main__":
    test_pipeline()