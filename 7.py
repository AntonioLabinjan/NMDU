import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 15

# --- LOGGING SETUP (Automatski folderi za svaki experiment) ---
def get_run_folder(base_name="run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"):
        n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()
print(f"\n🚀 Svi rezultati ovog treninga idu u: {RUN_DIR}")

# --- MODEL ---
class VoiceNetMedium(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- DATASET KLASA ---
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(20)
        self.time_mask = torchaudio.transforms.TimeMasking(40)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if self.augment:
            waveform = waveform + 0.002 * torch.randn_like(waveform)
            
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
        )(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        if self.augment:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)
            
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
        ).squeeze(0)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)
        return mel_spec, label

# --- PRIPREMA ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]

# Logiranje konfiguracije
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump({
        "lr": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": EPOCHS, 
        "weight_decay": WEIGHT_DECAY, "num_classes": len(class_names)
    }, f, indent=4)

all_files, all_labels = [], []
for label, folder in enumerate(person_folders):
    files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
    for f in files:
        all_files.append(f)
        all_labels.append(label)

X_train_f, X_temp_f, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
X_val_f, X_test_f, y_val, y_test = train_test_split(X_temp_f, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(VoiceDataset(X_train_f, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(VoiceDataset(X_val_f, y_val, augment=False), batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(VoiceDataset(X_test_f, y_test, augment=False), batch_size=BATCH_SIZE)

# --- TRENING SETUP ---
model = VoiceNetMedium(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')
patience_counter = 0

# --- TRENING LOOP ---
print(f"Trening započet na: {DEVICE}")
for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False)
    for bx, by in pbar:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * bx.size(0)
        t_corr += (out.argmax(1) == by).sum().item()
        t_total += bx.size(0)
    
    model.eval()
    v_loss, v_corr, v_total = 0, 0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx)
            loss = criterion(out, by)
            v_loss += loss.item() * bx.size(0)
            v_corr += (out.argmax(1) == by).sum().item()
            v_total += bx.size(0)

    # Statistika
    history['train_loss'].append(t_loss/t_total)
    history['val_loss'].append(v_loss/v_total)
    history['train_acc'].append(100*t_corr/t_total)
    history['val_acc'].append(100*v_corr/v_total)
    
    scheduler.step(v_loss/v_total)
    print(f"Epoch {epoch+1:03d} | Train Acc: {history['train_acc'][-1]:.1f}% | Train Loss: {history['train_loss'][-1]:.3f} |Val Acc: {history['val_acc'][-1]:.1f}% | Val Loss: {history['val_loss'][-1]:.3f}")

    # Best Model Save & Early Stopping
    if (v_loss/v_total) < best_val_loss:
        best_val_loss = v_loss/v_total
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[!] Early stopping na epohi {epoch+1}")
        break

# --- FINALNI REPORT ---
def save_reports():
    print("\nGeneriram izvještaje...")
    # 1. Krivulje
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.savefig(os.path.join(RUN_DIR, "training_curves.png"))

    # 2. Sanity check spektrograma (FIX: .squeeze())
    sample_spec, _ = next(iter(val_loader))
    plt.figure(figsize=(6, 4))
    plt.imshow(sample_spec[0].squeeze().cpu().numpy(), aspect='auto', origin='lower')
    plt.title("Model Input Sample")
    plt.savefig(os.path.join(RUN_DIR, "input_sample.png"))

    # 3. Test Evaluation
    model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth")))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(DEVICE))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(by.numpy())

    with open(os.path.join(RUN_DIR, "report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # 4. Normalizirana matrica
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))
    plt.close('all')

save_reports()
print(f"✅ Završeno. Rezultati su u: {RUN_DIR}")

(venv_voice311) antonio@antonio-IdeaPad-Slim-3-15IAH8:~/Desktop/NMDU$ python3 neural_network.py

🚀 Svi rezultati ovog treninga idu u: run_11
Trening započet na: cpu
Epoch 001 | Train Acc: 10.3% | Train Loss: 3.438 |Val Acc: 12.1% | Val Loss: 3.283          
Epoch 002 | Train Acc: 19.0% | Train Loss: 3.080 |Val Acc: 19.9% | Val Loss: 3.095          
Epoch 003 | Train Acc: 24.6% | Train Loss: 2.822 |Val Acc: 17.5% | Val Loss: 3.246          
Epoch 004 | Train Acc: 30.8% | Train Loss: 2.641 |Val Acc: 26.1% | Val Loss: 2.889          
Epoch 005 | Train Acc: 34.8% | Train Loss: 2.470 |Val Acc: 22.6% | Val Loss: 2.769          
Epoch 006 | Train Acc: 38.8% | Train Loss: 2.352 |Val Acc: 23.6% | Val Loss: 3.259          
Epoch 007 | Train Acc: 41.5% | Train Loss: 2.256 |Val Acc: 29.2% | Val Loss: 2.654          
Epoch 008 | Train Acc: 46.2% | Train Loss: 2.116 |Val Acc: 31.0% | Val Loss: 2.573          
Epoch 009 | Train Acc: 48.8% | Train Loss: 2.048 |Val Acc: 31.6% | Val Loss: 2.655          
Epoch 010 | Train Acc: 50.3% | Train Loss: 1.978 |Val Acc: 39.8% | Val Loss: 2.389          
Epoch 011 | Train Acc: 52.2% | Train Loss: 1.919 |Val Acc: 18.3% | Val Loss: 3.666          
Epoch 012 | Train Acc: 54.5% | Train Loss: 1.862 |Val Acc: 37.2% | Val Loss: 2.582          
Epoch 013 | Train Acc: 54.1% | Train Loss: 1.845 |Val Acc: 38.8% | Val Loss: 2.416          
Epoch 014 | Train Acc: 55.1% | Train Loss: 1.784 |Val Acc: 32.9% | Val Loss: 3.013          
Epoch 015 | Train Acc: 58.8% | Train Loss: 1.723 |Val Acc: 41.7% | Val Loss: 2.322          
Epoch 016 | Train Acc: 61.1% | Train Loss: 1.674 |Val Acc: 43.3% | Val Loss: 2.173          
Epoch 017 | Train Acc: 61.5% | Train Loss: 1.643 |Val Acc: 43.5% | Val Loss: 2.394          
Epoch 018 | Train Acc: 62.8% | Train Loss: 1.617 |Val Acc: 48.0% | Val Loss: 2.029          
Epoch 019 | Train Acc: 62.5% | Train Loss: 1.602 |Val Acc: 46.0% | Val Loss: 2.180          
Epoch 020 | Train Acc: 64.5% | Train Loss: 1.556 |Val Acc: 50.7% | Val Loss: 2.051          
Epoch 021 | Train Acc: 64.6% | Train Loss: 1.540 |Val Acc: 43.7% | Val Loss: 2.258          
Epoch 022 | Train Acc: 66.2% | Train Loss: 1.504 |Val Acc: 34.9% | Val Loss: 2.711          
Epoch 023 | Train Acc: 68.2% | Train Loss: 1.453 |Val Acc: 41.9% | Val Loss: 2.559          
Epoch 024 | Train Acc: 69.3% | Train Loss: 1.432 |Val Acc: 45.4% | Val Loss: 2.157          
Epoch 025 | Train Acc: 69.4% | Train Loss: 1.401 |Val Acc: 37.6% | Val Loss: 2.644          
Epoch 026 | Train Acc: 69.5% | Train Loss: 1.400 |Val Acc: 55.0% | Val Loss: 1.943          
Epoch 027 | Train Acc: 71.8% | Train Loss: 1.363 |Val Acc: 46.4% | Val Loss: 2.394          
Epoch 028 | Train Acc: 72.1% | Train Loss: 1.328 |Val Acc: 39.2% | Val Loss: 2.816          
Epoch 029 | Train Acc: 71.7% | Train Loss: 1.335 |Val Acc: 50.5% | Val Loss: 2.140          
Epoch 030 | Train Acc: 72.7% | Train Loss: 1.316 |Val Acc: 27.5% | Val Loss: 3.431          
Epoch 031 | Train Acc: 72.8% | Train Loss: 1.307 |Val Acc: 46.0% | Val Loss: 2.390          
Epoch 032 | Train Acc: 73.9% | Train Loss: 1.277 |Val Acc: 45.4% | Val Loss: 2.206          
Epoch 033 | Train Acc: 75.0% | Train Loss: 1.242 |Val Acc: 43.3% | Val Loss: 2.547          
Epoch 034 | Train Acc: 75.7% | Train Loss: 1.231 |Val Acc: 39.0% | Val Loss: 2.834          
Epoch 035 | Train Acc: 79.1% | Train Loss: 1.149 |Val Acc: 46.4% | Val Loss: 2.439          
Epoch 036 | Train Acc: 80.0% | Train Loss: 1.123 |Val Acc: 61.4% | Val Loss: 1.699          
Epoch 037 | Train Acc: 80.6% | Train Loss: 1.112 |Val Acc: 47.0% | Val Loss: 2.372          
Epoch 038 | Train Acc: 80.3% | Train Loss: 1.116 |Val Acc: 51.7% | Val Loss: 2.090          
Epoch 039 | Train Acc: 80.5% | Train Loss: 1.106 |Val Acc: 60.0% | Val Loss: 1.696          
Epoch 040 | Train Acc: 80.9% | Train Loss: 1.091 |Val Acc: 50.7% | Val Loss: 2.127          
Epoch 041 | Train Acc: 82.0% | Train Loss: 1.076 |Val Acc: 54.4% | Val Loss: 2.093          
Epoch 042 | Train Acc: 82.3% | Train Loss: 1.061 |Val Acc: 57.9% | Val Loss: 1.950          
Epoch 043 | Train Acc: 81.3% | Train Loss: 1.068 |Val Acc: 51.1% | Val Loss: 2.164          
Epoch 044 | Train Acc: 83.1% | Train Loss: 1.043 |Val Acc: 52.6% | Val Loss: 2.072          
Epoch 045 | Train Acc: 82.6% | Train Loss: 1.050 |Val Acc: 49.1% | Val Loss: 2.343          
Epoch 046 | Train Acc: 83.5% | Train Loss: 1.036 |Val Acc: 58.1% | Val Loss: 1.862          
Epoch 047 | Train Acc: 83.6% | Train Loss: 1.035 |Val Acc: 53.4% | Val Loss: 2.223          
Epoch 048 | Train Acc: 84.6% | Train Loss: 1.002 |Val Acc: 52.4% | Val Loss: 2.268          
Epoch 049 | Train Acc: 86.4% | Train Loss: 0.954 |Val Acc: 58.1% | Val Loss: 1.951          
Epoch 050 | Train Acc: 85.2% | Train Loss: 0.982 |Val Acc: 56.1% | Val Loss: 2.013          
Epoch 051 | Train Acc: 86.0% | Train Loss: 0.969 |Val Acc: 58.7% | Val Loss: 1.919          
Epoch 052 | Train Acc: 85.6% | Train Loss: 0.972 |Val Acc: 57.9% | Val Loss: 1.936          
Epoch 053 | Train Acc: 85.7% | Train Loss: 0.957 |Val Acc: 51.1% | Val Loss: 2.106          
Epoch 054 | Train Acc: 86.1% | Train Loss: 0.965 |Val Acc: 52.8% | Val Loss: 2.208          

[!] Early stopping na epohi 54

Generiram izvještaje...
✅ Završeno. Rezultati su u: run_11


              precision    recall  f1-score   support

     id10270       0.52      1.00      0.68        16
     id10271       0.38      0.86      0.52         7
     id10272       0.50      0.80      0.62         5
     id10273       0.39      0.88      0.54        24
     id10274       0.00      0.00      0.00         5
     id10275       0.67      0.57      0.62         7
     id10276       0.89      0.42      0.57        19
     id10277       0.50      0.67      0.57         6
     id10278       1.00      0.53      0.69        19
     id10279       0.50      0.33      0.40         6
     id10280       1.00      0.67      0.80         6
     id10281       0.36      0.50      0.42         8
     id10282       0.89      1.00      0.94         8
     id10283       0.52      0.71      0.60        24
     id10284       0.54      0.78      0.64         9
     id10285       0.71      1.00      0.83        10
     id10286       1.00      0.47      0.64        15
     id10287       1.00      0.50      0.67         4
     id10288       1.00      1.00      1.00         4
     id10289       0.67      0.50      0.57         8
     id10290       0.60      0.64      0.62        14
     id10291       0.83      0.71      0.77         7
     id10292       0.74      0.96      0.84        27
     id10293       0.88      0.70      0.78        20
     id10294       0.80      0.86      0.83        14
     id10295       0.67      0.22      0.33         9
     id10296       0.50      0.10      0.17        10
     id10297       1.00      0.25      0.40         8
     id10298       0.43      1.00      0.60        13
     id10299       0.00      0.00      0.00         5
     id10300       0.79      0.74      0.77        31
     id10301       0.00      0.00      0.00         4
     id10302       1.00      0.29      0.45        17
     id10303       1.00      0.73      0.84        11
     id10304       1.00      0.06      0.12        16
     id10305       0.57      0.57      0.57        14
     id10306       0.71      0.89      0.79        19
     id10307       0.90      0.56      0.69        16
     id10308       0.60      0.50      0.55         6
     id10309       0.67      0.71      0.69        17

    accuracy                           0.64       488
   macro avg       0.67      0.59      0.58       488
weighted avg       0.71      0.64      0.62       488
