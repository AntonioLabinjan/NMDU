"""
Blood Cell Classifier — GUI Inference Playground
=================================================
Single-file Tkinter GUI around the GAP-CNN blood cell classifier.

Features
--------
- "Random from TEST" button: picks a random image from your TEST_DIR,
  automatically reads the ground-truth label from the parent folder name,
  runs the model, and shows prediction + confidence bars next to the image.
- "Load Image..." button: pick any image file manually (no ground truth).
- Live class-probability bars (canvas-drawn, color-coded per class).
- Correct / Incorrect indicator when ground truth is known.
- Editable paths for dataset TEST dir and model checkpoint, with persistence
  to a small config file (gui_config.json) next to this script.

Usage
-----
    python blood_cell_gui.py

Requires: torch, torchvision, pillow, numpy  (tkinter ships with most
Python installs; on Linux you may need `sudo apt install python3-tk`)
"""

import json
import random
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ──────────────────────────────────────────────────────────────────────────
# Model definition (matches the training checkpoint)
# ──────────────────────────────────────────────────────────────────────────

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = 64
DEFAULT_CHECKPOINT = "/home/antonio/Desktop/NMDU_cells/Notebook_and_models_tweaked/best_gap_cnn_clean_tracking.pth"
DEFAULT_TEST_DIR = (
    "/home/antonio/.cache/kagglehub/datasets/paultimothymooney/blood-cells/"
    "versions/6/dataset2-master/dataset2-master/images/TEST"
)
CONFIG_PATH = Path(__file__).resolve().parent / "gui_config.json"

# Colors per class (hex, used both in tk and for bars)
CLASS_COLORS = {
    "EOSINOPHIL": "#e74c3c",
    "LYMPHOCYTE": "#2ecc71",
    "MONOCYTE": "#f39c12",
    "NEUTROPHIL": "#3498db",
}

BG = "#0f1117"
PANEL_BG = "#171a22"
FG = "#e8e8ec"
DIM_FG = "#8a8d98"
ACCENT = "#3498db"
GOOD = "#2ecc71"
BAD = "#e74c3c"


class GAPBloodCellCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]

        self.features = nn.Sequential(
            *conv_block(3, 16),
            *conv_block(16, 32),
            *conv_block(32, 64),
            *conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Identity(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device: torch.device) -> GAPBloodCellCNN:
    model = GAPBloodCellCNN(num_classes=len(CLASSES))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


def collect_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def ground_truth_from_path(image_path: Path, test_dir: Path):
    """Infer ground-truth class from the image's parent folder name."""
    try:
        rel_parts = image_path.relative_to(test_dir).parts
    except ValueError:
        rel_parts = image_path.parts
    for part in rel_parts:
        upper = part.upper()
        if upper in CLASSES:
            return upper
    # fallback: direct parent folder name
    parent = image_path.parent.name.upper()
    return parent if parent in CLASSES else None


# ──────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────

class BloodCellApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blood Cell Classifier — Inference Playground")
        self.geometry("980x640")
        self.configure(bg=BG)
        self.minsize(860, 560)

        self.model = None
        self.device = resolve_device()
        self.transform = build_transform()
        self.current_image_path = None
        self.tk_image = None  # keep ref

        self.cfg = self._load_config()

        self._build_style()
        self._build_layout()
        self._set_status(f"Ready. Device: {self.device}", DIM_FG)

    # ---------------------------------------------------------------- config
    def _load_config(self):
        defaults = {"checkpoint": DEFAULT_CHECKPOINT, "test_dir": DEFAULT_TEST_DIR}
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r") as f:
                    saved = json.load(f)
                defaults.update(saved)
            except Exception:
                pass
        return defaults

    def _save_config(self):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(
                    {
                        "checkpoint": self.checkpoint_var.get(),
                        "test_dir": self.test_dir_var.get(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    # ----------------------------------------------------------------- style
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL_BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Dim.TLabel", background=PANEL_BG, foreground=DIM_FG, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=DIM_FG, font=("Segoe UI", 9))
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )
        style.map("Accent.TButton", background=[("active", "#2980b9")])
        style.configure(
            "Secondary.TButton",
            background="#2a2e3a",
            foreground=FG,
            font=("Segoe UI", 10),
            padding=8,
        )
        style.map("Secondary.TButton", background=[("active", "#383d4d")])
        style.configure("TEntry", fieldbackground="#1c2030", foreground=FG, insertcolor=FG)

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        # Header
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=20, pady=(16, 8))
        ttk.Label(header, text="🩸 Blood Cell Classifier", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="GAP-CNN inference playground · EOSINOPHIL · LYMPHOCYTE · MONOCYTE · NEUTROPHIL",
            style="Sub.TLabel",
        ).pack(anchor="w")

        # Config row
        cfg_frame = ttk.Frame(self, style="TFrame")
        cfg_frame.pack(fill="x", padx=20, pady=(4, 10))

        ttk.Label(cfg_frame, text="Checkpoint:", style="TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.checkpoint_var = tk.StringVar(value=self.cfg["checkpoint"])
        ckpt_entry = ttk.Entry(cfg_frame, textvariable=self.checkpoint_var, width=46)
        ckpt_entry.grid(row=0, column=1, sticky="we", padx=(0, 6))
        ttk.Button(cfg_frame, text="Browse...", style="Secondary.TButton",
                   command=self._browse_checkpoint).grid(row=0, column=2, padx=(0, 16))

        self.load_btn = ttk.Button(cfg_frame, text="Load Model", style="Accent.TButton",
                                    command=self._load_model_clicked)
        self.load_btn.grid(row=0, column=3, padx=(0, 0))

        ttk.Label(cfg_frame, text="TEST dir:", style="TLabel").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(8, 0))
        self.test_dir_var = tk.StringVar(value=self.cfg["test_dir"])
        test_entry = ttk.Entry(cfg_frame, textvariable=self.test_dir_var, width=46)
        test_entry.grid(row=1, column=1, sticky="we", padx=(0, 6), pady=(8, 0))
        ttk.Button(cfg_frame, text="Browse...", style="Secondary.TButton",
                   command=self._browse_test_dir).grid(row=1, column=2, padx=(0, 16), pady=(8, 0))

        cfg_frame.columnconfigure(1, weight=1)

        # Action buttons
        action_frame = ttk.Frame(self, style="TFrame")
        action_frame.pack(fill="x", padx=20, pady=(0, 10))
        ttk.Button(action_frame, text="🎲 Random from TEST", style="Accent.TButton",
                   command=self._random_image_clicked).pack(side="left", padx=(0, 10))
        ttk.Button(action_frame, text="📁 Load Image...", style="Secondary.TButton",
                   command=self._load_image_clicked).pack(side="left")

        # Status bar
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self, textvariable=self.status_var, style="Sub.TLabel")
        self.status_label.pack(fill="x", padx=20, pady=(0, 8))

        # Main content: image (left) + results (right)
        content = ttk.Frame(self, style="TFrame")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # Left panel: image
        left = ttk.Frame(content, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.image_canvas = tk.Canvas(left, bg=PANEL_BG, highlightthickness=0, width=380, height=380)
        self.image_canvas.pack(expand=True, fill="both", padx=16, pady=16)
        self._draw_placeholder()

        self.path_label = ttk.Label(left, text="No image loaded", style="Dim.TLabel", wraplength=380)
        self.path_label.pack(fill="x", padx=16, pady=(0, 16))

        # Right panel: results
        right = ttk.Frame(content, style="Panel.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.gt_label = ttk.Label(right, text="Ground truth: —", style="Panel.TLabel", font=("Segoe UI", 12, "bold"))
        self.gt_label.pack(anchor="w", padx=16, pady=(16, 2))

        self.pred_label = ttk.Label(right, text="Prediction: —", style="Panel.TLabel", font=("Segoe UI", 12, "bold"))
        self.pred_label.pack(anchor="w", padx=16, pady=(2, 2))

        self.verdict_label = ttk.Label(right, text="", style="Panel.TLabel", font=("Segoe UI", 11, "bold"))
        self.verdict_label.pack(anchor="w", padx=16, pady=(2, 12))

        ttk.Label(right, text="Class probabilities", style="Dim.TLabel").pack(anchor="w", padx=16)

        self.bars_canvas = tk.Canvas(right, bg=PANEL_BG, highlightthickness=0, height=180)
        self.bars_canvas.pack(fill="x", padx=16, pady=(8, 16))

        # Running tally
        self.tally_label = ttk.Label(right, text="Session: 0 correct / 0 total", style="Dim.TLabel")
        self.tally_label.pack(anchor="w", padx=16, pady=(0, 16))

        self.correct_count = 0
        self.total_count = 0

        self.bind("<Configure>", lambda e: self._redraw_bars())
        self._last_probs = None
        self._last_pred = None

    # --------------------------------------------------------------- helpers
    def _draw_placeholder(self):
        self.image_canvas.delete("all")
        self.image_canvas.create_text(
            190, 190, text="No image yet\n(load model, then\npick a random image)",
            fill=DIM_FG, font=("Segoe UI", 11), justify="center"
        )

    def _set_status(self, text, color=FG):
        self.status_var.set(text)
        self.status_label.configure(foreground=color)

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")],
        )
        if path:
            self.checkpoint_var.set(path)

    def _browse_test_dir(self):
        path = filedialog.askdirectory(title="Select TEST dataset folder")
        if path:
            self.test_dir_var.set(path)

    # -------------------------------------------------------------- actions
    def _load_model_clicked(self):
        ckpt_path = self.checkpoint_var.get().strip()
        if not ckpt_path or not Path(ckpt_path).exists():
            messagebox.showerror("Checkpoint not found", f"Could not find checkpoint:\n{ckpt_path}")
            return
        self._save_config()
        self._set_status("Loading model...", ACCENT)
        self.load_btn.configure(state="disabled")

        def worker():
            try:
                model = load_model(ckpt_path, self.device)
                self.model = model
                self.after(0, lambda: self._set_status(
                    f"Model loaded ✓  (device: {self.device})", GOOD))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("Failed to load model", f"{e}\n\n{tb}"))
                self.after(0, lambda: self._set_status("Failed to load model.", BAD))
            finally:
                self.after(0, lambda: self.load_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _ensure_model(self):
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Click \"Load Model\" first.")
            return False
        return True

    def _random_image_clicked(self):
        if not self._ensure_model():
            return
        test_dir = Path(self.test_dir_var.get().strip())
        if not test_dir.exists():
            messagebox.showerror("TEST dir not found", f"Could not find folder:\n{test_dir}")
            return
        self._save_config()
        self._set_status("Scanning TEST folder...", ACCENT)

        def worker():
            try:
                images = collect_images(test_dir)
                if not images:
                    self.after(0, lambda: messagebox.showerror("No images", "No images found in TEST dir."))
                    self.after(0, lambda: self._set_status("No images found.", BAD))
                    return
                img_path = random.choice(images)
                gt = ground_truth_from_path(img_path, test_dir)
                self.after(0, lambda: self._run_inference(img_path, gt))
            except Exception as e:
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("Error", f"{e}\n\n{tb}"))

        threading.Thread(target=worker, daemon=True).start()

    def _load_image_clicked(self):
        if not self._ensure_model():
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        img_path = Path(path)
        test_dir = Path(self.test_dir_var.get().strip())
        gt = ground_truth_from_path(img_path, test_dir) if test_dir.exists() else None
        self._run_inference(img_path, gt)

    def _run_inference(self, img_path: Path, ground_truth):
        self._set_status(f"Running inference on {img_path.name}...", ACCENT)
        try:
            pil_img = Image.open(img_path).convert("RGB")
            tensor = self.transform(pil_img).unsqueeze(0)
            pred_class, confidence, probs = predict(self.model, tensor, self.device)
        except Exception as e:
            tb = traceback.format_exc()
            messagebox.showerror("Inference failed", f"{e}\n\n{tb}")
            self._set_status("Inference failed.", BAD)
            return

        self.current_image_path = img_path
        self._display_image(pil_img)
        self.path_label.configure(text=str(img_path))

        if ground_truth:
            self.gt_label.configure(
                text=f"Ground truth: {ground_truth}",
                foreground=CLASS_COLORS.get(ground_truth, FG),
            )
        else:
            self.gt_label.configure(text="Ground truth: unknown", foreground=DIM_FG)

        self.pred_label.configure(
            text=f"Prediction: {pred_class}  ({confidence * 100:.1f}%)",
            foreground=CLASS_COLORS.get(pred_class, FG),
        )

        if ground_truth:
            self.total_count += 1
            if pred_class == ground_truth:
                self.correct_count += 1
                self.verdict_label.configure(text="✓ Correct", foreground=GOOD)
            else:
                self.verdict_label.configure(text="✗ Incorrect", foreground=BAD)
            acc = self.correct_count / self.total_count * 100
            self.tally_label.configure(
                text=f"Session: {self.correct_count} correct / {self.total_count} total ({acc:.1f}%)"
            )
        else:
            self.verdict_label.configure(text="")

        self._last_probs = probs
        self._last_pred = pred_class
        self._redraw_bars()
        self._set_status(f"Done. Predicted {pred_class} ({confidence * 100:.1f}%).", GOOD)

    def _display_image(self, pil_img: Image.Image):
        self.image_canvas.update_idletasks()
        cw = max(self.image_canvas.winfo_width(), 320)
        ch = max(self.image_canvas.winfo_height(), 320)
        side = min(cw, ch) - 20
        side = max(side, 200)

        disp = pil_img.copy()
        disp.thumbnail((side, side), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(disp)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, image=self.tk_image, anchor="center")

    def _redraw_bars(self):
        canvas = self.bars_canvas
        canvas.delete("all")
        if self._last_probs is None:
            return

        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 300)
        n = len(CLASSES)
        row_h = 38
        top_pad = 10
        label_w = 110
        max_bar_w = width - label_w - 70

        canvas.configure(height=n * row_h + top_pad)

        for i, cls in enumerate(CLASSES):
            p = float(self._last_probs[i])
            y = top_pad + i * row_h
            color = CLASS_COLORS[cls]
            is_pred = cls == self._last_pred
            text_color = FG if is_pred else DIM_FG

            canvas.create_text(8, y + row_h / 2, text=cls, anchor="w",
                                fill=text_color, font=("Segoe UI", 10, "bold" if is_pred else "normal"))

            bar_x0 = label_w
            bar_y0 = y + 8
            bar_y1 = y + row_h - 10
            bar_x1 = bar_x0 + max_bar_w
            canvas.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, fill="#22263033", outline="")
            fill_w = bar_x0 + max_bar_w * p
            bar_color = color if is_pred else self._dim_color(color)
            canvas.create_rectangle(bar_x0, bar_y0, fill_w, bar_y1, fill=bar_color, outline="")

            canvas.create_text(bar_x1 + 8, y + row_h / 2, text=f"{p * 100:5.1f}%",
                                anchor="w", fill=text_color, font=("Segoe UI", 10))

    @staticmethod
    def _dim_color(hex_color: str, factor: float = 0.45) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"


def main():
    app = BloodCellApp()
    app.mainloop()


if __name__ == "__main__":
    main()
