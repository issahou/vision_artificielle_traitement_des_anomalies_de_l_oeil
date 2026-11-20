# view/image_canvas.py
"""
Widget d'affichage d'une image OpenCV dans Tkinter.
Conversion BGR → RGB → PIL → ImageTk avec redimensionnement préservant le ratio.
"""
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImageCanvas(tk.Frame):

    def __init__(self, parent, title: str, width: int = 300, height: int = 280):
        super().__init__(parent, bd=1, relief=tk.SUNKEN)
        self.width = width
        self.height = height
        self._photo = None

        tk.Label(self, text=title, font=("Arial", 10, "bold"),
                 bg="#f0f0f0").pack(fill="x")
        self.label_image = tk.Label(self, bg="black")
        self.label_image.pack(expand=True, fill=tk.BOTH)

    def display(self, image_bgr):
        if image_bgr is None:
            self.clear()
            return

        if len(image_bgr.shape) == 2:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        scale = min(self.width / w, self.height / h, 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        rgb_resized = cv2.resize(rgb, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(rgb_resized)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.label_image.configure(image=self._photo)

    def clear(self):
        self.label_image.configure(image="")
        self._photo = None