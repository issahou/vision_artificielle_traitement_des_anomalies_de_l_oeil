# main.py
"""
Point d'entrée. Lance l'application.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller.app_controller import AppController


def main():
    # ⚠️ ADAPTER CE CHEMIN vers ton dataset E-ophtha-EX
    DATASET_ROOT = os.path.join("data", "e_ophtha_EX")

    if not os.path.exists(DATASET_ROOT):
        print(f"⚠️  Attention : dataset introuvable dans {DATASET_ROOT}")
        print("   Vérifie le chemin ou place tes images dans data/e_ophtha_EX/EX/")
        # On continue quand même : chargement manuel possible via le bouton
        DATASET_ROOT = "data"

    app = AppController(DATASET_ROOT)
    app.run()


if __name__ == "__main__":
    main()