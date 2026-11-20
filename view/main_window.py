# view/main_window.py
"""
Fenêtre principale Tkinter. Aucune logique métier.
Expose des callbacks que le Controller branche.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Optional, Dict, Any
from view.image_canvas import ImageCanvas
from view.param_panel import ParamPanel
from view.metrics_panel import MetricsPanel


class MainWindow(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("G10 — Détection d'Exsudats (E-ophtha-EX)")
        self.geometry("1500x950")

        self.on_load_image: Optional[Callable] = None
        self.on_load_gt: Optional[Callable] = None
        self.on_params_changed: Optional[Callable] = None
        self.on_save_results: Optional[Callable] = None
        self.on_sweep_threshold: Optional[Callable] = None
        self.on_quit: Optional[Callable] = None

        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=5)
        top.pack(fill="x")

        ttk.Button(top, text="📂 Charger image",
                   command=lambda: self._call(self.on_load_image)).pack(side="left", padx=3)
        ttk.Button(top, text="🎯 Charger GT",
                   command=lambda: self._call(self.on_load_gt)).pack(side="left", padx=3)
        ttk.Button(top, text="💾 Sauvegarder",
                   command=lambda: self._call(self.on_save_results)).pack(side="left", padx=3)
        ttk.Button(top, text="📈 Balayer seuils (PR)",
                   command=lambda: self._call(self.on_sweep_threshold)).pack(side="left", padx=3)

        self.info = ttk.Label(top, text="Aucune image chargée", foreground="#666")
        self.info.pack(side="right", padx=10)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=5, pady=5)

        self.param_panel = ParamPanel(main, on_change=self._params_changed)
        main.add(self.param_panel, weight=0)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        grid = ttk.Frame(right)
        grid.pack(fill="both", expand=True)

        self.canvases: Dict[str, ImageCanvas] = {}
        layout = [
            ("original", "Originale"),
            ("preprocessed", "Prétraitée (CLAHE)"),
            ("morphology", "Top-hat (exsudats)"),
            ("pred_mask", "Masque prédit"),
            ("overlay", "Overlay (rouge = détection)"),
            ("gt_mask", "Masque GT (vérité)"),
        ]
        for i, (key, title) in enumerate(layout):
            r, c = i // 3, i % 3
            cv = ImageCanvas(grid, title, 380, 280)
            cv.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
            self.canvases[key] = cv
        for i in range(2):
            grid.rowconfigure(i, weight=1)
        for j in range(3):
            grid.columnconfigure(j, weight=1)

        self.metrics_panel = MetricsPanel(right)
        self.metrics_panel.pack(fill="x", pady=5)

    @staticmethod
    def _call(cb):
        if cb:
            cb()

    def _params_changed(self):
        if self.on_params_changed:
            self.on_params_changed()

    # API pour le Controller
    def get_params(self) -> Dict[str, Any]:
        return self.param_panel.get_params()

    def display_results(self, results: Dict[str, Any]):
        for key, canvas in self.canvases.items():
            img = results.get(key)
            if img is not None:
                canvas.display(img)
            else:
                canvas.clear()
        self.metrics_panel.update_metrics(results.get("metrics", {}))

    def set_info(self, text: str):
        self.info.config(text=text)

    def ask_image_path(self) -> str:
        return filedialog.askopenfilename(
            title="Choisir une image EX",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("Tous", "*.*")])

    def ask_gt_path(self) -> str:
        return filedialog.askopenfilename(
            title="Choisir le masque GT",
            filetypes=[("PNG", "*.png"), ("Tous", "*.*")])

    def show_error(self, msg: str):
        messagebox.showerror("Erreur", msg)

    def show_info(self, msg: str):
        messagebox.showinfo("Info", msg)