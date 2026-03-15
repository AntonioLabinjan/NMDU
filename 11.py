# NAS experiments => we have a new best locally
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import numpy as np
import optuna
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# --- POSTAVKE LOGIRANJA ---
logging.getLogger("optuna").setLevel(logging.WARNING)

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 48
N_TRIALS = 30 
MAX_EPOCHS_PER_TRIAL = 15

# --- DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            
            mel_spec = self.mel_transform(waveform)
            mel_spec = self.db_transform(mel_spec)
            mel_spec = torch.nn.functional.interpolate(
                mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
            ).squeeze(0)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)
            return mel_spec, label
        except Exception:
            return torch.zeros((1, 128, 128)), label

# --- DYNAMIC MODEL ---
class DynamicVoiceNet(nn.Module):
    def __init__(self, trial, num_classes):
        super().__init__()
        layers = []
        n_layers = trial.suggest_int("n_layers", 3, 6)
        in_channels = 1
        for i in range(n_layers):
            out_channels = trial.suggest_int(f"units_l{i}", 16, 128, step=16)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- NAS OBJECTIVE ---
def objective(trial):
    current_best = "N/A"
    try:
        current_best = f"{trial.study.best_value:.4f}"
    except Exception: pass

    print(f"\n\033[94m{'━'*70}\033[0m")
    print(f"🚀 \033[1mTRIALS #{trial.number}/{N_TRIALS}\033[0m | \033[92mBEST SO FAR: {current_best}\033[0m")
    
    person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
    class_names = [os.path.basename(f) for f in person_folders]
    all_files, all_labels = [], []
    for label, folder in enumerate(person_folders):
        for f in glob.glob(os.path.join(folder, "**/*.wav"), recursive=True):
            all_files.append(f); all_labels.append(label)

    X_train, X_val, y_train, y_val = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels)
    train_loader = DataLoader(VoiceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(VoiceDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = DynamicVoiceNet(trial, len(class_names)).to(DEVICE)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"⚙️  \033[2mLR: {lr:.6f} | Layers: {trial.params['n_layers']} | Dropout: {trial.params['dropout']:.2f}\033[0m")

    for epoch in range(MAX_EPOCHS_PER_TRIAL):
        # --- TRENING FAZA ---
        model.train()
        train_loss, t_corr, t_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"   Ep {epoch+1:02d}/{MAX_EPOCHS_PER_TRIAL}", leave=False, bar_format='{l_bar}{bar:20}{r_bar}')
        
        for bx, by in pbar:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            t_corr += (out.argmax(1) == by).sum().item()
            t_total += bx.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # --- VALIDACIJSKA FAZA ---
        model.eval()
        val_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                v_loss = criterion(out, by)
                val_loss += v_loss.item()
                v_corr += (out.argmax(1) == by).sum().item()
                v_total += bx.size(0)
        
        # Izračun finalnih metrika za epohu
        avg_t_loss = train_loss / len(train_loader)
        avg_v_loss = val_loss / len(val_loader)
        train_acc = t_corr / t_total
        val_acc = v_corr / v_total

        # Verbose ispis sa svim traženim metrikama
        print(f"   📊 Ep {epoch+1:02d} -> "
              f"T-Loss: \033[90m{avg_t_loss:.4f}\033[0m | T-Acc: \033[90m{train_acc:.4f}\033[0m | "
              f"V-Loss: \033[95m{avg_v_loss:.4f}\033[0m | V-Acc: \033[1m\033[92m{val_acc:.4f}\033[0m")

        # Optuna Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"   \033[91m❌ PRUNED\033[0m at epoch {epoch+1}")
            raise optuna.exceptions.TrialPruned()

    return val_acc

if __name__ == "__main__":
    print("\n\033[1m" + "="*70)
    print("  VOICE RECOGNITION - ULTIMATE VERBOSE NAS")
    print("="*70 + "\033[0m")
    
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )
    
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\n\033[93m⚠️ Prekinuto. Analiziram najbolje rezultate...\033[0m")

    if len(study.trials) > 0:
        print("\n" + "\033[1m\033[92m" + "═"*70)
        print("🏆 POBJEDNIČKA ARHITEKTURA 🏆")
        print(f"Finalna Točnost: {study.best_value:.4f}")
        print("\nOptimalni parametri:")
        for key, value in study.best_params.items():
            print(f" 🔹 {key}: {value}")
        print("═"*70 + "\033[0m")
