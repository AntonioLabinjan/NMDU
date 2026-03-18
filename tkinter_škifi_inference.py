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
from collections import Counter
import time

# --- CONFIG ---
MODEL_PATH = "triplet_run_2/best_triplet_model.pth"
INDEX_NAME = "voice_database.index"
METADATA_NAME = "voice_metadata.pkl"
SAMPLE_RATE = 16000
CHUNK = 1024 * 16 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAD_THRESHOLD = 0.005
ENROLL_DURATION = 60  # Sekunde snimanja za novu osobu
NUM_SEGMENTS = 15     # Na koliko komada režemo snimku

class VoiceNetEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        def conv_block(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_f),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
        self.features = torch.nn.Sequential(
            conv_block(1, 32), conv_block(32, 64), conv_block(64, 128),
            conv_block(128, 256), conv_block(256, 256)
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

class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice ID: Pro Voting System 🚀")
        self.root.geometry("600x800")
        
        self.pa = pyaudio.PyAudio()
        self.model = VoiceNetEmbedding().to(DEVICE)
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        
        self.load_database()
        self.is_running = False
        self.top_k = tk.IntVar(value=15) # Povećan k zbog više vektora po osobi
        self.threshold = 0.75
        
        self.setup_ui()

    def load_database(self):
        if os.path.exists(INDEX_NAME) and os.path.exists(METADATA_NAME):
            self.index = faiss.read_index(INDEX_NAME)
            with open(METADATA_NAME, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(128)
            self.metadata = []

    def get_mic_devices(self):
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                devices.append(f"{i}: {info.get('name')}")
        return devices

    def setup_ui(self):
        ttk.Label(self.root, text="Sustav Prepoznavanja Glasa v2", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        mic_frame = ttk.LabelFrame(self.root, text="Odabir Mikrofona")
        mic_frame.pack(pady=5, fill="x", padx=20)
        self.mic_selector = ttk.Combobox(mic_frame, values=self.get_mic_devices(), state="readonly", width=50)
        self.mic_selector.pack(pady=10, padx=10)
        if self.get_mic_devices(): self.mic_selector.current(len(self.get_mic_devices())-1)

        self.pred_label = ttk.Label(self.root, text="STATUS: ČEKAM...", font=("Helvetica(311)", 14), foreground="gray")
        self.pred_label.pack(pady=15)

        settings_frame = ttk.LabelFrame(self.root, text="Postavke Pretrage")
        settings_frame.pack(pady=5, fill="x", padx=20)
        ttk.Label(settings_frame, text="Top-K:").grid(row=0, column=0, padx=10, pady=5)
        ttk.Entry(settings_frame, textvariable=self.top_k, width=8).grid(row=0, column=1)

        self.btn_toggle = tk.Button(self.root, text="START MIKROFON", bg="#4CAF50", fg="white", font=("Helvetica", 11, "bold"), command=self.toggle_mic)
        self.btn_toggle.pack(pady=10, ipadx=20)
        
        enroll_frame = ttk.LabelFrame(self.root, text="Registracija (60s duboka analiza)")
        enroll_frame.pack(pady=10, fill="x", padx=20)
        self.new_name = tk.StringVar()
        ttk.Entry(enroll_frame, textvariable=self.new_name, width=20).pack(side="left", padx=10, pady=10)
        self.btn_enroll = ttk.Button(enroll_frame, text="Započni Enrollment", command=self.enroll_voice)
        self.btn_enroll.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        self.log_box = tk.Text(self.root, height=12, width=75, font=("Consolas", 9))
        self.log_box.pack(pady=10, padx=10)

    def log(self, message):
        self.log_box.insert(tk.END, f"> {message}\n")
        self.log_box.see(tk.END)

    def toggle_mic(self):
        if not self.is_running:
            try:
                self.selected_mic_id = int(self.mic_selector.get().split(":")[0])
                self.is_running = True
                self.btn_toggle.config(text="STOP MIKROFON", bg="#f44336")
                threading.Thread(target=self.mic_loop, daemon=True).start()
                self.log(f"Inference pokrenut na mic ID {self.selected_mic_id}")
            except:
                messagebox.showerror("Greška", "Odaberi mikrofon!")
        else:
            self.is_running = False
            self.btn_toggle.config(text="START MIKROFON", bg="#4CAF50")

    def preprocess_audio(self, audio_np):
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        if waveform.shape[1] < 1024: return None # Premalo za FFT
        mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
        db_tf = torchaudio.transforms.AmplitudeToDB()
        spec = db_tf(mel_tf(waveform))
        spec = F.interpolate(spec.unsqueeze(0), size=(128, 128)).to(DEVICE)
        spec = (spec - spec.mean()) / (spec.std() + 1e-7)
        return spec

    def mic_loop(self):
        stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, 
                              input_device_index=self.selected_mic_id, frames_per_buffer=CHUNK)
        while self.is_running:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if np.max(np.abs(audio_np)) < VAD_THRESHOLD:
                    self.root.after(0, lambda: self.pred_label.config(text="STATUS: TIŠINA", foreground="gray"))
                    continue

                spec = self.preprocess_audio(audio_np)
                if spec is None: continue

                with torch.inference_mode():
                    emb = self.model(spec).cpu().numpy().astype('float32')
                
                if self.index.ntotal > 0:
                    k_val = min(self.top_k.get(), self.index.ntotal)
                    distances, indices = self.index.search(emb, k_val)
                    
                    # WEIGHTED VOTING LOGIKA
                    weighted_votes = {}
                    for d, idx in zip(distances[0], indices[0]):
                        if d < self.threshold:
                            name = self.metadata[idx]['person_id']
                            weight = 1.0 / (d + 1e-6) # Bliži vektori vrijede više
                            weighted_votes[name] = weighted_votes.get(name, 0) + weight
                    
                    if weighted_votes:
                        winner = max(weighted_votes, key=weighted_votes.get)
                        total_w = sum(weighted_votes.values())
                        conf = (weighted_votes[winner] / total_w) * 100
                        self.root.after(0, lambda w=winner, c=conf: self.pred_label.config(
                            text=f"OSOBA: {w} ({c:.1f}% siguran)", foreground="#2E7D32"))
                    else:
                        self.root.after(0, lambda: self.pred_label.config(text="STATUS: NEPOZNATO", foreground="#C62828"))
            except: break
        stream.stop_stream(); stream.close()

    def enroll_voice(self):
        name = self.new_name.get().strip()
        if not name:
            messagebox.showwarning("Ime?", "Moraš unijeti ime za registraciju.")
            return
            
        def record_and_process():
            self.root.after(0, lambda: self.btn_enroll.config(state="disabled"))
            self.log(f"Započinjem snimanje 60s za {name}. Pričaj normalno...")
            
            try:
                mic_id = int(self.mic_selector.get().split(":")[0])
                stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, 
                                      input_device_index=mic_id, frames_per_buffer=CHUNK)
                
                all_audio = []
                total_chunks = int(SAMPLE_RATE / CHUNK * ENROLL_DURATION)
                
                for i in range(total_chunks):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    all_audio.append(data)
                    if i % 10 == 0:
                        progress_val = (i / total_chunks) * 100
                        self.root.after(0, lambda v=progress_val: self.progress.config(value=v))
                
                stream.stop_stream(); stream.close()
                self.log("Snimanje gotovo. Obrađujem vektore...")
                
                full_audio = np.frombuffer(b''.join(all_audio), dtype=np.int16).astype(np.float32) / 32768.0
                
                # REZANJE NA 15 SEGMENATA + SPREMANJE
                seg_size = len(full_audio) // NUM_SEGMENTS
                added_count = 0
                
                for i in range(NUM_SEGMENTS):
                    segment = full_audio[i*seg_size : (i+1)*seg_size]
                    # Provjeri je li segment tišina (da ne punimo bazu smećem)
                    if np.max(np.abs(segment)) > VAD_THRESHOLD:
                        spec = self.preprocess_audio(segment)
                        if spec is not None:
                            with torch.inference_mode():
                                emb = self.model(spec).cpu().numpy().astype('float32')
                            self.index.add(emb)
                            self.metadata.append({"person_id": name, "file_path": f"LIVE_SEG_{i}"})
                            added_count += 1
                
                faiss.write_index(self.index, INDEX_NAME)
                with open(METADATA_NAME, 'wb') as f:
                    pickle.dump(self.metadata, f)
                
                self.log(f"Registracija završena! Dodano {added_count} vektora za {name}.")
                self.root.after(0, lambda: self.progress.config(value=0))
                self.root.after(0, lambda: self.btn_enroll.config(state="normal"))
                self.new_name.set("")
                
            except Exception as e:
                self.log(f"Greška kod snimanja: {e}")
                self.root.after(0, lambda: self.btn_enroll.config(state="normal"))

        threading.Thread(target=record_and_process, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()
