import torch
import torchaudio
import torch.nn.functional as F
import os
import faiss
import numpy as np
import pickle
import pyaudio
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import math
import time

# --- CONFIG ---
MODEL_PATH = "triplet_run_2/best_triplet_model.pth"
INDEX_NAME = "voice_database.index"
METADATA_NAME = "voice_metadata.pkl"
SAMPLE_RATE = 16000
CHUNK = 1024 * 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAD_THRESHOLD = 0.005
ENROLL_DURATION = 60
NUM_SEGMENTS = 15

# --- PALETTE ---
BG_MAIN    = "#05070d"
BG_CARD    = "#0d1117"
BG_CARD2   = "#0a0e18"
ACCENT     = "#4cc9f0"
ACCENT2    = "#4361ee"
DANGER     = "#f72585"
SUCCESS    = "#4cc9f0"
TEXT_MAIN  = "#f3f4f6"
TEXT_MUTED = "#64748b"
BORDER     = "#1a2035"
BORDER_HI  = "#4cc9f0"

CLUSTER_COLORS = ["#4cc9f0", "#4361ee", "#f72585", "#7209b7", "#3a0ca3", "#4895ef", "#560bad"]

FONT_MONO  = ("Courier New", 10)
FONT_TITLE = ("Helvetica Neue", 11, "bold")
FONT_HERO  = ("Helvetica Neue", 28, "bold")
FONT_LABEL = ("Helvetica Neue", 9)
FONT_BTN   = ("Helvetica Neue", 10, "bold")
FONT_BADGE = ("Courier New", 8, "bold")

# --- MODEL ---
class VoiceNetEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        def conv_block(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_f, out_f, 3, padding=1),
                torch.nn.BatchNorm2d(out_f),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
        self.features = torch.nn.Sequential(
            conv_block(1, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256), conv_block(256, 256)
        )
        self.embedding_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedding_head(x)
        return F.normalize(x, p=2, dim=1)

# --- PCA Embedding Canvas ---
class EmbeddingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_CARD2, highlightthickness=1, 
                         highlightbackground=BORDER, **kwargs)
        self.points = [] 

    def update_embeddings(self, embeddings, labels):
        if len(embeddings) < 2: return
        X = np.array(embeddings)
        X_centered = X - np.mean(X, axis=0)
        u, s, vh = np.linalg.svd(X_centered, full_matrices=False)
        X_pca = u[:, :2] * s[:2] 
        x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
        y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
        x_range = (x_max - x_min) if x_max != x_min else 1
        y_range = (y_max - y_min) if y_max != y_min else 1
        self.points = []
        unique_labels = list(set(labels))
        for i in range(len(X_pca)):
            nx = (X_pca[i, 0] - x_min) / x_range
            ny = (X_pca[i, 1] - y_min) / y_range
            color_idx = unique_labels.index(labels[i]) % len(CLUSTER_COLORS)
            self.points.append((nx, ny, CLUSTER_COLORS[color_idx]))
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10: return
        pad = 25
        for nx, ny, color in self.points:
            px = pad + nx * (w - 2 * pad)
            py = pad + ny * (h - 2 * pad)
            self.create_oval(px-4, py-4, px+4, py+4, fill=color, outline=BG_MAIN, width=1)

# --- Waveform Canvas ---
class WaveformCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_MAIN, highlightthickness=0, **kwargs)
        self.active   = False
        self.phase    = 0.0
        self.amp      = 0.0          
        self._job     = None
        self._animate()

    def set_active(self, val: bool):
        self.active = val
        if not val: self.amp = 0.0

    def set_amplitude(self, a: float):
        self.amp = min(1.0, max(0.0, a))

    def _animate(self):
        self._draw()
        self._job = self.after(33, self._animate)          

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2: return
        cy = h / 2
        waves = [(ACCENT, 0.55, 0.0), (ACCENT2, 0.35, 0.9)]
        for color, base_alpha, offset in waves:
            if not self.active:
                pts = []
                for x in range(0, w + 4, 4): pts += [x, cy]
                if len(pts) >= 4: self.create_line(*pts, fill=self._dim(color, 0.18), smooth=True, width=1)
                continue
            pts = []
            for x in range(0, w + 4, 4):
                t = x / w
                amp = cy * 0.55 * self.amp
                y = cy + amp * math.sin(2 * math.pi * (t * 3 + self.phase + offset))
                y += amp * 0.4 * math.sin(2 * math.pi * (t * 7 - self.phase * 1.3 + offset))
                pts += [x, y]
            if len(pts) >= 4: self.create_line(*pts, fill=color, smooth=True, width=2 if color == ACCENT else 1)
        self.phase += 0.035

    @staticmethod
    def _dim(hex_color: str, alpha: float) -> str:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        r2, g2, b2 = int(r * alpha + 5 * (1 - alpha)), int(g * alpha + 7 * (1 - alpha)), int(b * alpha + 13 * (1 - alpha))
        return f"#{r2:02x}{g2:02x}{b2:02x}"

def card(parent, label="", pady=(14, 14)):
    outer = tk.Frame(parent, bg=BORDER, bd=0)
    outer.pack(fill="x", pady=6)
    inner = tk.Frame(outer, bg=BG_CARD, bd=0)
    inner.pack(fill="x", padx=1, pady=1)
    if label:
        tk.Label(inner, text=label.upper(), bg=BG_CARD, fg=TEXT_MUTED, font=FONT_BADGE).pack(anchor="w", padx=16, pady=(10, 0))
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=0, pady=(6, 0))
    body = tk.Frame(inner, bg=BG_CARD)
    body.pack(fill="x", padx=16, pady=pady)
    return body

class FlatButton(tk.Label):
    def __init__(self, parent, text, command=None, bg=ACCENT, fg="black", hover_bg=None, **kwargs):
        self._bg, self._fg, self._hover, self._command = bg, fg, hover_bg or self._lighten(bg), command
        super().__init__(parent, text=text, bg=bg, fg=fg, font=FONT_BTN, cursor="hand2", pady=10, padx=0, anchor="center", relief="flat", bd=0, **kwargs)
        self.bind("<Enter>", lambda _: self.config(bg=self._hover))
        self.bind("<Leave>", lambda _: self.config(bg=self._bg))
        self.bind("<Button-1>", self._on_click)
    def _on_click(self, _): 
        if self._command: self._command()
    @staticmethod
    def _lighten(hex_color: str) -> str:
        r, g, b = min(255, int(hex_color[1:3], 16) + 30), min(255, int(hex_color[3:5], 16) + 30), min(255, int(hex_color[5:7], 16) + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

class StatusBadge(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=BG_MAIN, **kwargs)
        self._dot = tk.Label(self, text="●", bg=BG_MAIN, fg=TEXT_MUTED, font=("Courier New", 10))
        self._label = tk.Label(self, text="IDLE", bg=BG_MAIN, fg=TEXT_MUTED, font=FONT_BADGE)
        self._dot.pack(side="left"); self._label.pack(side="left", padx=(4, 0))
    def set(self, text, color):
        self._dot.config(fg=color); self._label.config(text=text.upper(), fg=color)

# --- MAIN APP ---
class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptryx Echo")
        self.root.geometry("580x1050")
        self.root.configure(bg=BG_MAIN)
        self.pa = pyaudio.PyAudio()
        self.model = VoiceNetEmbedding().to(DEVICE)
        self._load_model(); self._load_db()

        self.is_running = False
        self.top_k = tk.IntVar(value=15)
        self.threshold = tk.DoubleVar(value=0.75) # Changed to DoubleVar for tuning

        self._build_ui(); self._pulse_loop()
        self.root.after(1000, self.update_plot)

    def _load_model(self):
        if not os.path.exists(MODEL_PATH): return
        sd = torch.load(MODEL_PATH, map_location=DEVICE)
        try: self.model.load_state_dict(sd)
        except: self.model.load_state_dict({k.replace("embedding_head", "embedding_head"): v for k, v in sd.items()})
        self.model.eval()

    def _load_db(self):
        if os.path.exists(INDEX_NAME) and os.path.exists(METADATA_NAME):
            self.index = faiss.read_index(INDEX_NAME)
            with open(METADATA_NAME, 'rb') as f: self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(128); self.metadata = []

    def _build_ui(self):
        topbar = tk.Frame(self.root, bg=BG_CARD, height=48)
        topbar.pack(fill="x"); topbar.pack_propagate(False)
        tk.Label(topbar, text="PERCEPTRYX", bg=BG_CARD, fg=ACCENT, font=("Courier New", 11, "bold")).pack(side="left", padx=18)
        self.status_badge = StatusBadge(topbar); self.status_badge.pack(side="right", padx=18)

        canvas = tk.Canvas(self.root, bg=BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg=BG_MAIN)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=560)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=10); scrollbar.pack(side="right", fill="y")
        
        body = self.scrollable_frame
        hero = tk.Frame(body, bg=BG_CARD2, highlightthickness=1, highlightbackground=BORDER)
        hero.pack(fill="x", pady=(10, 6))
        self.pred_label = tk.Label(hero, text="——", bg=BG_CARD2, fg=TEXT_MUTED, font=("Courier New", 34, "bold"), pady=10)
        self.pred_label.pack()
        self.conf_label = tk.Label(hero, text="WAITING FOR AUDIO", bg=BG_CARD2, fg=TEXT_MUTED, font=FONT_BADGE)
        self.conf_label.pack(pady=(0, 8))

        self.wave = WaveformCanvas(body, height=60); self.wave.pack(fill="x", pady=6)

        mic_body = card(body, "Microphone")
        self.mic_var = tk.StringVar(); devs = self._get_mics()
        cb = ttk.Combobox(mic_body, textvariable=self.mic_var, values=devs, state="readonly", font=FONT_MONO)
        cb.pack(fill="x")
        if devs: cb.current(len(devs)-1)

        self.btn_toggle = FlatButton(body, "▶  START LISTENING", command=self.toggle_mic, bg=ACCENT, fg="#05070d")
        self.btn_toggle.pack(fill="x", pady=10)

        # --- TUNER CARD ---
        tune_box = card(body, "Inference Settings")
        # K Tuner
        k_row = tk.Frame(tune_box, bg=BG_CARD)
        k_row.pack(fill="x", pady=5)
        tk.Label(k_row, text="K NEIGHBORS:", bg=BG_CARD, fg=TEXT_MUTED, font=FONT_BADGE).pack(side="left")
        tk.Spinbox(k_row, from_=1, to=100, textvariable=self.top_k, bg=BG_CARD2, fg=ACCENT, bd=0, font=FONT_MONO, width=5).pack(side="right")
        # Threshold Tuner
        t_row = tk.Frame(tune_box, bg=BG_CARD)
        t_row.pack(fill="x", pady=5)
        tk.Label(t_row, text="THRESHOLD:", bg=BG_CARD, fg=TEXT_MUTED, font=FONT_BADGE).pack(side="left")
        tk.Scale(t_row, from_=0.1, to=1.5, resolution=0.01, orient="horizontal", variable=self.threshold, bg=BG_CARD, highlightthickness=0, troughcolor=BG_CARD2, fg=ACCENT).pack(side="right", fill="x", expand=True, padx=(10,0))

        pca_box = card(body, "Latent Space Analysis (PCA)")
        self.pca_canvas = EmbeddingCanvas(pca_box, height=200); self.pca_canvas.pack(fill="x")
        tk.Label(pca_box, text="Clusters represent distinct voice signatures", bg=BG_CARD, fg=TEXT_MUTED, font=FONT_BADGE).pack(pady=(5,0))

        en = card(body, "Enrollment")
        self.new_name = tk.StringVar(value="Enter name...")
        tk.Entry(en, textvariable=self.new_name, bg=BG_CARD2, fg=TEXT_MAIN, insertbackground=ACCENT, bd=0, font=FONT_MONO).pack(fill="x", pady=(0, 8), ipady=8)
        self.btn_enroll = FlatButton(en, "⬤  START ENROLLMENT", command=self.enroll_voice, bg=ACCENT2, fg=TEXT_MAIN)
        self.btn_enroll.pack(fill="x")

        self.log_box = tk.Text(body, height=6, bg=BG_CARD2, fg=ACCENT, bd=0, font=FONT_MONO)
        self.log_box.pack(fill="x", pady=10)

    def update_plot(self):
        if self.index.ntotal > 1:
            embeddings = [self.index.reconstruct(i) for i in range(self.index.ntotal)]
            labels = [self.metadata[i]["person_id"] for i in range(self.index.ntotal)]
            self.pca_canvas.update_embeddings(embeddings, labels)

    def log(self, msg):
        self.log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n"); self.log_box.see("end")

    def _get_mics(self):
        devs = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0: devs.append(f"{i}: {info['name']}")
        return devs

    def toggle_mic(self):
        if not self.is_running:
            try: self.selected_mic_id = int(self.mic_var.get().split(":")[0])
            except: return
            self.is_running = True; self.wave.set_active(True)
            self.btn_toggle.config(text="■  STOP LISTENING", bg=DANGER, fg=TEXT_MAIN)
            threading.Thread(target=self._mic_loop, daemon=True).start()
        else:
            self.is_running = False; self.wave.set_active(False)
            self.btn_toggle.config(text="▶  START LISTENING", bg=ACCENT, fg="#05070d")

    def preprocess_audio(self, audio_np):
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        if waveform.shape[1] < 1024: return None
        mel = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
        spec = torchaudio.transforms.AmplitudeToDB()(mel(waveform))
        spec = F.interpolate(spec.unsqueeze(0), size=(128, 128)).to(DEVICE)
        spec = (spec - spec.mean()) / (spec.std() + 1e-7)
        return spec

    def _mic_loop(self):
        stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, input_device_index=self.selected_mic_id, frames_per_buffer=CHUNK)
        while self.is_running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.root.after(0, lambda r=float(np.sqrt(np.mean(audio**2))): self.wave.set_amplitude(r * 20))
                if np.max(np.abs(audio)) < VAD_THRESHOLD: continue
                spec = self.preprocess_audio(audio)
                if spec is None: continue
                with torch.inference_mode(): emb = self.model(spec).cpu().numpy().astype("float32")
                if self.index.ntotal > 0:
                    # Dynamically use the slider/spinbox values
                    k_val = min(self.top_k.get(), self.index.ntotal)
                    thresh_val = self.threshold.get()
                    
                    D, I = self.index.search(emb, k_val)
                    votes = {}
                    for d, idx in zip(D[0], I[0]):
                        if d < thresh_val:
                            name = self.metadata[idx]["person_id"]
                            votes[name] = votes.get(name, 0) + (1.0 / (d + 1e-6))
                    if votes:
                        winner = max(votes, key=votes.get)
                        self.root.after(0, lambda w=winner: self._set_pred(w, ACCENT, "IDENTIFIED"))
                    else:
                        self.root.after(0, lambda: self._set_pred("UNKNOWN", DANGER, "NO MATCH"))
            except: break
        stream.close()

    def _set_pred(self, name, color, sub=""):
        self.pred_label.config(text=name, fg=color)
        self.conf_label.config(text=sub, fg=color if sub else TEXT_MUTED)

    def enroll_voice(self):
        name = self.new_name.get().strip()
        if not name or name == "Enter name...": return
        def run():
            self.log(f"Recording {name}...")
            stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, input_device_index=self.selected_mic_id, frames_per_buffer=CHUNK)
            chunks = []
            for _ in range(int(SAMPLE_RATE / CHUNK * ENROLL_DURATION)): chunks.append(stream.read(CHUNK))
            stream.close()
            audio = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0
            seg_size = len(audio) // NUM_SEGMENTS
            for i in range(NUM_SEGMENTS):
                spec = self.preprocess_audio(audio[i*seg_size:(i+1)*seg_size])
                if spec is not None:
                    with torch.inference_mode(): emb = self.model(spec).cpu().numpy().astype("float32")
                    self.index.add(emb); self.metadata.append({"person_id": name})
            faiss.write_index(self.index, INDEX_NAME)
            with open(METADATA_NAME, "wb") as f: pickle.dump(self.metadata, f)
            self.root.after(0, self.update_plot); self.log(f"✓ {name} Enrolled.")
        threading.Thread(target=run, daemon=True).start()

    def _pulse_loop(self):
        if self.is_running: self.status_badge.set("LIVE", ACCENT if int(time.time() * 2) % 2 == 0 else TEXT_MUTED)
        self.root.after(500, self._pulse_loop)

if __name__ == "__main__":
    root = tk.Tk(); app = VoiceApp(root); root.mainloop()
