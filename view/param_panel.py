# view/param_panel.py
"""
Panneau de paramètres avec sliders/menus/checkboxes dans un canvas scrollable.
Chaque widget déclenche un callback on_change (branché au Controller).
"""
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any


class ParamPanel(tk.Frame):

    def __init__(self, parent, on_change: Callable[[], None]):
        super().__init__(parent, width=320)
        self.on_change = on_change
        self.vars: Dict[str, tk.Variable] = {}

        canvas = tk.Canvas(self, width=320, bg="#fafafa")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build()

    def _section(self, title: str):
        ttk.Label(self.inner, text=title, font=("Arial", 10, "bold"),
                  foreground="#003366").pack(fill="x", padx=5, pady=(10, 2))

    def _slider(self, label: str, key: str, from_: float, to: float,
                default: float, resolution: float = 1):
        frame = ttk.Frame(self.inner)
        frame.pack(fill="x", padx=5, pady=1)
        ttk.Label(frame, text=label, width=18).pack(side="left")
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        scale = ttk.Scale(frame, from_=from_, to=to, variable=var,
                          orient="horizontal", length=120)
        scale.pack(side="left", fill="x", expand=True)
        val_label = ttk.Label(frame, text=f"{default:g}", width=7)
        val_label.pack(side="left")

        def update(*_):
            val_label.config(text=f"{var.get():.2f}")
            self.on_change()

        var.trace_add("write", update)

    def _combo(self, label: str, key: str, values: list, default: str):
        frame = ttk.Frame(self.inner)
        frame.pack(fill="x", padx=5, pady=1)
        ttk.Label(frame, text=label, width=18).pack(side="left")
        var = tk.StringVar(value=default)
        self.vars[key] = var
        cb = ttk.Combobox(frame, textvariable=var, values=values,
                          state="readonly", width=15)
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e: self.on_change())

    def _check(self, label: str, key: str, default: bool):
        frame = ttk.Frame(self.inner)
        frame.pack(fill="x", padx=5, pady=1)
        var = tk.BooleanVar(value=default)
        self.vars[key] = var
        ttk.Checkbutton(frame, text=label, variable=var,
                        command=self.on_change).pack(side="left")

    def _build(self):
        self._section("Chargement")
        self._combo("Canal", "channel",
                    ["green", "red", "blue", "gray"], "green")
        self._check("Recadrer sur disque", "crop", True)

        self._section("Prétraitement")
        self._slider("Flou illumination", "illum_blur_size", 15, 151, 71, 2)
        self._check("CLAHE activé", "clahe_enabled", True)
        self._slider("CLAHE clip", "clahe_clip", 0.5, 8.0, 3.0, 0.1)
        self._slider("CLAHE tile", "clahe_tile", 4, 16, 8)
        self._combo("Type flou", "blur_type",
                    ["median", "gaussian", "none"], "median")
        self._slider("Taille flou", "blur_size", 1, 15, 3, 2)
        self._slider("Unsharp force", "unsharp_strength", 0.0, 3.0, 0.5, 0.1)

        self._section("Morphologie (EX : top-hat)")
        self._combo("Type", "morpho_type",
                    ["tophat", "blackhat", "none"], "tophat")
        self._slider("Rayon SE", "morpho_radius", 1, 30, 15)
        self._slider("Itérations", "morpho_iterations", 1, 3, 1)
        self._slider("Nettoyage", "cleanup_size", 0, 3, 1)

        self._section("Segmentation")
        self._combo("Méthode seuil", "threshold_method",
                    ["otsu", "fixed", "adaptive"], "otsu")
        self._slider("Offset Otsu", "otsu_offset", -50, 50, 0, 2)
        self._slider("Seuil fixe", "fixed_threshold", 0, 255, 30)
        self._slider("Bloc adaptatif", "adaptive_block", 3, 51, 15, 2)
        self._slider("Constante C", "adaptive_C", -10, 10, 2)
        self._slider("Aire min", "min_area", 0, 200, 20)
        self._slider("Aire max", "max_area", 100, 10000, 5000, 50)
        self._slider("Circularité min", "min_circularity", 0.0, 1.0, 0.2, 0.05)
        self._slider("Ouverture post", "open_post", 0, 5, 1)
        self._slider("Fermeture post", "close_post", 0, 5, 1)

    def get_params(self) -> Dict[str, Any]:
        return {k: v.get() for k, v in self.vars.items()}