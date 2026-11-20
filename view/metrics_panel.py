# view/metrics_panel.py
import tkinter as tk
from tkinter import ttk
from typing import Dict


class MetricsPanel(ttk.LabelFrame):

    def __init__(self, parent, title: str = "Métriques"):
        super().__init__(parent, text=title, padding=5)
        self.labels: Dict[str, ttk.Label] = {}

        items = [
            ("precision", "Précision (px)"),
            ("recall", "Rappel (px)"),
            ("f1", "F1 (px)"),
            ("precision_obj", "Précision (obj)"),
            ("recall_obj", "Rappel (obj)"),
            ("f1_obj", "F1 (obj)"),
            ("n_pred_obj", "Objets prédits"),
            ("n_gt_obj", "Objets GT"),
        ]
        for i, (key, text) in enumerate(items):
            row, col = i // 4, i % 4
            ttk.Label(self, text=text + " :").grid(
                row=row, column=col * 2, sticky="e", padx=3, pady=2)
            lbl = ttk.Label(self, text="—", font=("Arial", 9, "bold"),
                            foreground="#003366")
            lbl.grid(row=row, column=col * 2 + 1, sticky="w", padx=3, pady=2)
            self.labels[key] = lbl

    def update_metrics(self, metrics: Dict[str, float]):
        if not metrics:
            for lbl in self.labels.values():
                lbl.config(text="—")
            return
        for key, lbl in self.labels.items():
            if key in metrics:
                v = metrics[key]
                if isinstance(v, float):
                    lbl.config(text=f"{v:.3f}")
                else:
                    lbl.config(text=str(v))