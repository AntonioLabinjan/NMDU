# best so far
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001  # Brže učenje za početak
WEIGHT_DECAY = 0.01    # Blaža regularizacija

# --- MODEL (VoiceNet Medium - Pojačana verzija) ---
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
        
        # Proširena arhitektura: 32 -> 64 -> 128 -> 128
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4), # Smanjen dropout da ne "guši" učenje
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
        # SpecAugment parametri
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=40)

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
            
        # Dodavanje šuma samo u treningu
        if self.augment:
            waveform = waveform + 0.002 * torch.randn_like(waveform)

        # Povećan broj melsa na 128 za bolju rezoluciju glasa
        mel_spec = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_mels=128)(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        if self.augment:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)
            
        # Standardizacija veličine slike na 128x128
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Normalizacija spektrograma
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)
        return mel_spec, label

# --- PRIPREMA PODATAKA ---
print(f"--- START: {datetime.now().strftime('%H:%M:%S')} ---")
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]

all_files, all_labels = [], []
for label, folder in enumerate(person_folders):
    files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
    for f in files:
        all_files.append(f)
        all_labels.append(label)

X_train_f, X_temp_f, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
X_val_f, X_test_f, y_val, y_test = train_test_split(X_temp_f, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(VoiceDataset(X_train_f, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VoiceDataset(X_val_f, y_val, augment=False), batch_size=BATCH_SIZE)
test_loader = DataLoader(VoiceDataset(X_test_f, y_test, augment=False), batch_size=BATCH_SIZE)

print(f"[DATA] Klasa: {len(class_names)} | Train: {len(X_train_f)} | Val: {len(X_val_f)}")

# --- TRENING SETUP ---
model = VoiceNetMedium(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # Smanjen smoothing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)

best_val_loss = float('inf')

# --- LOOP ---
for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [TRAIN]", unit="batch", leave=False)
    
    for bx, by in train_pbar:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        
        t_loss += loss.item() * bx.size(0)
        t_corr += (out.argmax(1) == by).sum().item()
        t_total += bx.size(0)
        train_pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*t_corr/t_total:.1f}%"})
    
    model.eval()
    v_loss, v_corr, v_total = 0, 0, 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [VAL]", unit="batch", leave=False)
    
    with torch.no_grad():
        for bx, by in val_pbar:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx)
            loss = criterion(out, by)
            v_loss += loss.item() * bx.size(0)
            v_corr += (out.argmax(1) == by).sum().item()
            v_total += bx.size(0)
            val_pbar.set_postfix({"v_loss": f"{loss.item():.3f}", "v_acc": f"{100*v_corr/v_total:.1f}%"})

    train_l, train_a = t_loss/t_total, t_corr/t_total*100
    val_l, val_a = v_loss/v_total, v_corr/v_total*100
    
    scheduler.step(val_l)
    
    print(f"Epoch {epoch+1:03d} | L: {train_l:.4f}, A: {train_a:.2f}% | VL: {val_l:.4f}, VA: {val_a:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if val_l < best_val_loss:
        best_val_loss = val_l
        torch.save(model.state_dict(), "best_medium_model.pth")
        print(f"  [+] New Best Model: {best_val_loss:.4f}")

# --- EVALUACIJA ---
print(f"\n[TESTING] Učitavam najbolji model...")
model.load_state_dict(torch.load("best_medium_model.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for bx, by in tqdm(test_loader, desc="Final Eval"):
        out = model(bx.to(DEVICE))
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(by.numpy())

print("\n" + "="*40 + "\n IZVJEŠTAJ \n" + "="*40)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

(venv_voice311) antonio@antonio-IdeaPad-Slim-3-15IAH8:~/Desktop/NMDU$ python3 neural_network.py
--- START: 15:13:13 ---
[DATA] Klasa: 40 | Train: 3899 | Val: 487
Epoch 001/80 [TRAIN]:   0%|                                     | 0/122 [00:00<?, ?batch/s]/home/antonio/Desktop/NMDU/venv_voice311/lib/python3.11/site-packages/torchaudio/functional/functional.py:584: UserWarning: At least one mel filterbank has all zero values. The value for `n_mels` (128) may be set too high. Or, the value for `n_freqs` (201) may be set too low.
  warnings.warn(
Epoch 001 | L: 3.4867, A: 8.80% | VL: 3.4677, VA: 10.68% | LR: 0.001000                    
  [+] New Best Model: 3.4677
Epoch 002 | L: 3.1148, A: 17.34% | VL: 3.2151, VA: 12.53% | LR: 0.001000                   
  [+] New Best Model: 3.2151
Epoch 003 | L: 2.9038, A: 21.70% | VL: 3.0457, VA: 21.56% | LR: 0.001000                   
  [+] New Best Model: 3.0457
Epoch 004 | L: 2.7495, A: 26.37% | VL: 2.8665, VA: 20.94% | LR: 0.001000                   
  [+] New Best Model: 2.8665
Epoch 005 | L: 2.6016, A: 30.85% | VL: 2.5620, VA: 28.75% | LR: 0.001000                   
  [+] New Best Model: 2.5620
Epoch 006/80 [TRAIN]:   6%|▎     | 7/122 [00:06<01:51,  1.03batch/s, loss=Epoch 006/80 [TRAIN]:   6%|▎     | 7/122 [00:07<01:51,  1.03batch/s, loss=Epoch 006/80 [TRAIN]:   7%|▍     | 8Epoch 006 | L: 2.4983, A: 32.83% | VL: 3.4865, VA: 13.55% | LR: 0.001000                    
Epoch 007 | L: 2.4119, A: 36.86% | VL: 2.5494, VA: 30.60% | LR: 0.001000                    
  [+] New Best Model: 2.5494
Epoch 008 | L: 2.3451, A: 37.52% | VL: 3.1856, VA: 26.90% | LR: 0.001000                    
Epoch 009 | L: 2.2829, A: 40.52% | VL: 2.8525, VA: 29.57% | LR: 0.001000                    
Epoch 010 | L: 2.2276, A: 41.40% | VL: 2.8831, VA: 29.98% | LR: 0.001000                    
Epoch 011 | L: 2.1576, A: 44.52% | VL: 2.5478, VA: 31.62% | LR: 0.001000                    
  [+] New Best Model: 2.5478
Epoch 012 | L: 2.1037, A: 45.73% | VL: 2.8471, VA: 32.03% | LR: 0.001000                    
Epoch 013 | L: 2.0547, A: 46.63% | VL: 2.2921, VA: 44.15% | LR: 0.001000                    
  [+] New Best Model: 2.2921
Epoch 014 | L: 2.0365, A: 47.68% | VL: 2.9058, VA: 30.60% | LR: 0.001000                    
Epoch 015 | L: 2.0007, A: 48.35% | VL: 2.5966, VA: 36.55% | LR: 0.001000                    
Epoch 016 | L: 1.9532, A: 50.04% | VL: 2.2243, VA: 40.25% | LR: 0.001000                    
  [+] New Best Model: 2.2243
Epoch 017 | L: 1.9118, A: 51.19% | VL: 2.8463, VA: 35.73% | LR: 0.001000                    
Epoch 018 | L: 1.8687, A: 53.37% | VL: 2.2106, VA: 43.53% | LR: 0.001000                    
  [+] New Best Model: 2.2106
Epoch 019 | L: 1.8250, A: 54.83% | VL: 2.2216, VA: 49.08% | LR: 0.001000                    
Epoch 020 | L: 1.8291, A: 54.66% | VL: 2.1860, VA: 45.59% | LR: 0.001000                    
  [+] New Best Model: 2.1860
Epoch 021 | L: 1.7945, A: 55.89% | VL: 2.5732, VA: 38.60% | LR: 0.001000                    
Epoch 022 | L: 1.7474, A: 56.91% | VL: 2.5694, VA: 40.86% | LR: 0.001000                    
Epoch 023 | L: 1.7325, A: 57.99% | VL: 2.8429, VA: 39.43% | LR: 0.001000                    
Epoch 024 | L: 1.6989, A: 59.14% | VL: 2.6310, VA: 40.25% | LR: 0.001000                    
Epoch 025 | L: 1.6744, A: 59.43% | VL: 2.3809, VA: 46.00% | LR: 0.001000                    
Epoch 026 | L: 1.6409, A: 60.48% | VL: 1.9886, VA: 52.36% | LR: 0.001000                    
  [+] New Best Model: 1.9886
Epoch 027 | L: 1.6334, A: 61.58% | VL: 1.8993, VA: 56.47% | LR: 0.001000                    
  [+] New Best Model: 1.8993
Epoch 028 | L: 1.6221, A: 61.27% | VL: 2.6933, VA: 42.92% | LR: 0.001000                    
Epoch 029/80 [TRAIN]:  98%|████▉| 120/122 [01:24<00:01,  1.43batch/s, loss=1.539, acc=62.5%]Epoch 029/80 [VAL]:   6%|▍     | 1/16 [00:00<00:06,  2.39batch/s, v_loss=2.377, v_acc=53.1%]Epoch 029 | L: 1.5825, A: 62.48% | VL: 2.4477, VA: 42.51% | LR: 0.001000                    
Epoch 030 | L: 1.5519, A: 64.12% | VL: 2.7808, VA: 44.35% | LR: 0.001000                    
Epoch 031 | L: 1.5422, A: 64.09% | VL: 1.9385, VA: 55.24% | LR: 0.001000                    
Epoch 032 | L: 1.5200, A: 64.94% | VL: 2.1429, VA: 55.24% | LR: 0.001000                    
Epoch 033 | L: 1.5048, A: 65.81% | VL: 2.1291, VA: 50.72% | LR: 0.001000                    
Epoch 034 | L: 1.4751, A: 66.45% | VL: 2.1095, VA: 51.75% | LR: 0.001000                    
Epoch 035 | L: 1.4560, A: 66.76% | VL: 1.7586, VA: 59.55% | LR: 0.001000                    
  [+] New Best Model: 1.7586
Epoch 036 | L: 1.4468, A: 67.56% | VL: 3.3508, VA: 32.24% | LR: 0.001000                    
Epoch 037/80 [TRAIN]:   6%|▍      | 7/122 [00:04<01:21,  1.41batch/s, loss=1.643, acc=66.5%]Epoch 037 | L: 1.4430, A: 67.71% | VL: 2.3843, VA: 42.51% | LR: 0.001000                    
Epoch 038 | L: 1.4138, A: 68.81% | VL: 2.4576, VA: 46.41% | LR: 0.001000                    
Epoch 039 | L: 1.4031, A: 69.09% | VL: 2.2345, VA: 50.51% | LR: 0.001000                    
Epoch 040 | L: 1.3956, A: 69.22% | VL: 2.2113, VA: 51.33% | LR: 0.001000                    
Epoch 041 | L: 1.3764, A: 70.04% | VL: 1.9634, VA: 52.57% | LR: 0.001000                    
Epoch 042 | L: 1.3764, A: 70.27% | VL: 2.4066, VA: 47.23% | LR: 0.001000                    
Epoch 043 | L: 1.3374, A: 71.56% | VL: 2.3786, VA: 51.54% | LR: 0.000500                    
Epoch 044 | L: 1.2506, A: 74.51% | VL: 2.1558, VA: 56.06% | LR: 0.000500                    
Epoch 045 | L: 1.2398, A: 75.28% | VL: 1.7927, VA: 59.96% | LR: 0.000500                    
Epoch 046 | L: 1.2417, A: 75.63% | VL: 2.1687, VA: 54.00% | LR: 0.000500                    
Epoch 047 | L: 1.2174, A: 75.99% | VL: 1.8830, VA: 59.75% | LR: 0.000500                    
Epoch 048 | L: 1.1973, A: 77.02% | VL: 1.8118, VA: 60.37% | LR: 0.000500                    
Epoch 049 | L: 1.1882, A: 77.84% | VL: 1.8422, VA: 60.16% | LR: 0.000500                    
Epoch 050 | L: 1.1970, A: 76.25% | VL: 2.2703, VA: 51.95% | LR: 0.000500                    
Epoch 051 | L: 1.1845, A: 77.10% | VL: 2.1018, VA: 54.62% | LR: 0.000250                    
Epoch 052 | L: 1.1259, A: 79.48% | VL: 1.7723, VA: 61.19% | LR: 0.000250                    
Epoch 053 | L: 1.1173, A: 79.51% | VL: 1.7041, VA: 63.45% | LR: 0.000250                    
  [+] New Best Model: 1.7041
Epoch 054 | L: 1.1476, A: 79.07% | VL: 1.7933, VA: 61.81% | LR: 0.000250                    
Epoch 055 | L: 1.1360, A: 79.69% | VL: 1.8663, VA: 60.99% | LR: 0.000250                    
Epoch 056 | L: 1.1101, A: 80.41% | VL: 1.9357, VA: 58.32% | LR: 0.000250                    
Epoch 057 | L: 1.1107, A: 79.97% | VL: 1.8273, VA: 62.01% | LR: 0.000250                    
Epoch 058 | L: 1.1090, A: 80.43% | VL: 1.7388, VA: 61.19% | LR: 0.000250                    
Epoch 059 | L: 1.0850, A: 81.25% | VL: 1.7234, VA: 61.60% | LR: 0.000250                    
Epoch 060 | L: 1.0685, A: 82.48% | VL: 1.7553, VA: 60.57% | LR: 0.000250                    
Epoch 061 | L: 1.0960, A: 80.92% | VL: 1.6151, VA: 64.68% | LR: 0.000250                    
  [+] New Best Model: 1.6151
Epoch 062 | L: 1.0856, A: 81.12% | VL: 1.6934, VA: 64.68% | LR: 0.000250                    
Epoch 063 | L: 1.0761, A: 81.59% | VL: 1.8346, VA: 59.96% | LR: 0.000250                    
Epoch 064 | L: 1.0762, A: 81.38% | VL: 1.8634, VA: 58.93% | LR: 0.000250                    
Epoch 065 | L: 1.0832, A: 81.33% | VL: 1.9730, VA: 58.11% | LR: 0.000250                    
Epoch 066 | L: 1.0739, A: 82.28% | VL: 1.7804, VA: 61.19% | LR: 0.000250                    
Epoch 067 | L: 1.0658, A: 81.56% | VL: 1.6802, VA: 62.83% | LR: 0.000250                    
Epoch 068 | L: 1.0536, A: 82.33% | VL: 1.7135, VA: 62.42% | LR: 0.000250                    
Epoch 069 | L: 1.0608, A: 82.43% | VL: 1.7594, VA: 60.57% | LR: 0.000125                    
Epoch 070 | L: 1.0425, A: 83.33% | VL: 1.7019, VA: 63.86% | LR: 0.000125                    
Epoch 071 | L: 1.0297, A: 83.18% | VL: 1.8296, VA: 60.78% | LR: 0.000125                    
Epoch 072 | L: 1.0404, A: 83.07% | VL: 1.7876, VA: 61.81% | LR: 0.000125                    
Epoch 073 | L: 1.0316, A: 83.46% | VL: 1.7084, VA: 63.24% | LR: 0.000125                    
Epoch 074 | L: 1.0158, A: 84.51% | VL: 1.6050, VA: 64.27% | LR: 0.000125                    
  [+] New Best Model: 1.6050
Epoch 075 | L: 1.0319, A: 83.46% | VL: 1.5969, VA: 65.91% | LR: 0.000125                    
  [+] New Best Model: 1.5969
Epoch 076 | L: 1.0373, A: 83.15% | VL: 1.6995, VA: 64.48% | LR: 0.000125                    
Epoch 077 | L: 1.0213, A: 83.82% | VL: 1.6433, VA: 65.09% | LR: 0.000125                    
Epoch 078 | L: 1.0202, A: 83.71% | VL: 1.6748, VA: 64.68% | LR: 0.000125                    
Epoch 079 | L: 1.0117, A: 84.23% | VL: 1.8130, VA: 59.96% | LR: 0.000125                    
Epoch 080 | L: 1.0052, A: 84.94% | VL: 1.7760, VA: 62.63% | LR: 0.000125                    

[TESTING] Učitavam najbolji model...
Final Eval: 100%|███████████████████████████████████████████| 16/16 [00:08<00:00,  1.80it/s]

========================================
 IZVJEŠTAJ 
========================================
              precision    recall  f1-score   support

     id10270       0.93      0.88      0.90        16
     id10271       0.75      0.86      0.80         7
     id10272       0.60      0.60      0.60         5
     id10273       0.71      0.83      0.77        24
     id10274       0.40      0.40      0.40         5
     id10275       1.00      0.29      0.44         7
     id10276       0.82      0.74      0.78        19
     id10277       0.50      0.17      0.25         6
     id10278       0.47      0.79      0.59        19
     id10279       0.67      0.33      0.44         6
     id10280       0.60      0.50      0.55         6
     id10281       0.50      0.25      0.33         8
     id10282       1.00      0.88      0.93         8
     id10283       0.61      0.71      0.65        24
     id10284       0.86      0.67      0.75         9
     id10285       0.90      0.90      0.90        10
     id10286       0.93      0.87      0.90        15
     id10287       0.50      0.75      0.60         4
     id10288       1.00      0.25      0.40         4
     id10289       0.47      0.88      0.61         8
     id10290       0.70      0.50      0.58        14
     id10291       0.67      0.86      0.75         7
     id10292       0.59      1.00      0.74        27
     id10293       0.82      0.90      0.86        20
     id10294       0.71      0.86      0.77        14
     id10295       0.58      0.78      0.67         9
     id10296       0.00      0.00      0.00        10
     id10297       0.71      0.62      0.67         8
     id10298       0.62      0.62      0.62        13
     id10299       0.00      0.00      0.00         5
     id10300       0.85      0.55      0.67        31
     id10301       0.00      0.00      0.00         4
     id10302       0.89      0.47      0.62        17
     id10303       0.60      0.82      0.69        11
     id10304       0.75      0.19      0.30        16
     id10305       0.44      0.57      0.50        14
     id10306       0.86      0.95      0.90        19
     id10307       0.81      0.81      0.81        16
     id10308       0.57      0.67      0.62         6
     id10309       0.50      0.76      0.60        17

    accuracy                           0.68       488
   macro avg       0.65      0.61      0.60       488
weighted avg       0.69      0.68      0.65       488

