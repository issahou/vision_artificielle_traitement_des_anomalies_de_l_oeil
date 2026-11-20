# config/default_params.py
"""
Paramètres par défaut — calibrés pour E-ophtha-EX (exsudats = lésions CLAIRES).
Un seul dict centralisé : modifier ici = modifier partout.
"""

DEFAULT_PARAMS = {
    # --- Chargement ---
    "channel": "green",           # green / red / blue / gray
    "mask_threshold": 15,         # seuil pour détecter le disque rétinien
    "crop": True,                 # recadrer sur le disque

    # --- Prétraitement ---
    "illum_blur_size": 71,        # taille flou illumination (impair)
    "clahe_enabled": True,
    "clahe_clip": 3.0,            # EX : contraste un peu plus fort que MA
    "clahe_tile": 8,
    "blur_type": "median",        # median / gaussian / none
    "blur_size": 3,               # noyau flou (impair)
    "unsharp_strength": 0.5,      # EX : unsharp aide (exsudats = hautes fréquences)

    # --- Morphologie (CRUCIAL pour EX) ---
    "morpho_type": "tophat",      # tophat OBLIGATOIRE pour exsudats clairs
    "morpho_radius": 15,          # SE plus grand que MA (exsudats plus gros)
    "morpho_iterations": 1,
    "cleanup_size": 1,            # ouverture légère post-tophat

    # --- Segmentation ---
    "threshold_method": "otsu",
    "otsu_offset": 0,             # décalage (négatif = plus strict)
    "fixed_threshold": 30,
    "adaptive_block": 15,
    "adaptive_C": 2,
    "min_area": 20,               # EX : aire minimale plus grande que MA
    "max_area": 5000,             # EX : aire maximale plus grande
    "min_circularity": 0.2,       # EX : exsudats plus irréguliers
    "open_post": 1,
    "close_post": 1,
}