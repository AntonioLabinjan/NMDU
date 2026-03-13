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
BATCH_SIZE = 48  # Povećano radi stabilnosti gradijenta
EPOCHS = 100     # Dajemo mu malo više vremena
LEARNING_RATE = 0.0008 # Malo sporiji start
WEIGHT_DECAY = 0.02    # Jača kazna za prevelike težine
EARLY_STOP_PATIENCE = 20

def get_run_folder(base_name="run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"): n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()
print(f"\n🚀 POKRETANJE FINO-TUNINGA: Rezultati u {RUN_DIR}")

# --- MODEL (Dublja verzija s 5 slojeva) ---
class VoiceNetDeep(nn.Module):
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
            conv_block(128, 256), # Novi dublji sloj
            conv_block(256, 256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- DATASET (Balansirana Augmentacija) ---
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        # Smanjeno maskiranje da model ne "oslijepi"
        self.freq_mask = torchaudio.transforms.FrequencyMasking(20) 
        self.time_mask = torchaudio.transforms.TimeMasking(35)

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        waveform, sr = torchaudio.load(path)
        
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
            
        if self.augment:
            # Time shift ostavljamo - on je zlata vrijedan
            shift = int(waveform.shape[1] * 0.1)
            shift_val = np.random.randint(-shift, shift)
            waveform = torch.roll(waveform, shift_val, dims=1)
            # Smanjen šum da ne uništi vokalne karakteristike
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
        
        # Obavezna normalizacija
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)
        return mel_spec, label

# --- PRIPREMA ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]

all_files, all_labels = [], []
for label, folder in enumerate(person_folders):
    for f in glob.glob(os.path.join(folder, "**/*.wav"), recursive=True):
        all_files.append(f); all_labels.append(label)

X_train_f, X_temp_f, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
X_val_f, X_test_f, y_val, y_test = train_test_split(X_temp_f, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(VoiceDataset(X_train_f, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(VoiceDataset(X_val_f, y_val, augment=False), batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(VoiceDataset(X_test_f, y_test, augment=False), batch_size=BATCH_SIZE)

# --- TRENING ---
model = VoiceNetDeep(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
best_v_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1:03d}", leave=False)
    for bx, by in pbar:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad(); out = model(bx); loss = criterion(out, by)
        loss.backward(); optimizer.step()
        t_loss += loss.item()*bx.size(0); t_corr += (out.argmax(1)==by).sum().item(); t_total += bx.size(0)
    
    model.eval()
    v_loss, v_corr, v_total = 0, 0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx); loss = criterion(out, by)
            v_loss += loss.item()*bx.size(0); v_corr += (out.argmax(1)==by).sum().item(); v_total += bx.size(0)

    tl, ta = t_loss/t_total, 100*t_corr/t_total
    vl, va = v_loss/v_total, 100*v_corr/v_total
    history['t_loss'].append(tl); history['v_loss'].append(vl); history['t_acc'].append(ta); history['v_acc'].append(va)
    
    scheduler.step(vl)
    print(f"Ep {epoch+1:03d} | Train: {ta:.1f}% | Val: {va:.1f}% | Loss V: {vl:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if vl < best_v_loss:
        best_v_loss = vl
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print("\n[!] Prekidamo - nema više napretka."); break

# --- FINALNI REPORT ---
print("\n📊 Čuvam rezultate u " + RUN_DIR)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(history['t_loss'], label='T'); plt.plot(history['v_loss'], label='V'); plt.title('Loss'); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history['t_acc'], label='T'); plt.plot(history['v_acc'], label='V'); plt.title('Acc'); plt.legend()
plt.savefig(os.path.join(RUN_DIR, "training_curves.png"))

model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth")))
model.eval()
y_t, y_p = [], []
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(DEVICE))
        y_p.extend(out.argmax(1).cpu().numpy()); y_t.extend(by.numpy())

with open(os.path.join(RUN_DIR, "report.txt"), "w") as f:
    f.write(classification_report(y_t, y_p, target_names=class_names))

cm = confusion_matrix(y_t, y_p)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))

print(f"Gotovo!!")

Ep 001 | Train: 14.6% | Val: 12.1% | Loss V: 3.4010 | LR: 0.000800                          
Ep 002 | Train: 28.5% | Val: 24.4% | Loss V: 3.0709 | LR: 0.000800                          
Ep 003 | Train: 36.9% | Val: 26.3% | Loss V: 2.9670 | LR: 0.000800                          
Ep 004 | Train: 45.8% | Val: 31.4% | Loss V: 2.7243 | LR: 0.000800                          
Ep 005 | Train: 54.4% | Val: 29.0% | Loss V: 2.9378 | LR: 0.000800                          
Ep 006 | Train: 60.5% | Val: 41.9% | Loss V: 2.3422 | LR: 0.000800                          
Ep 007 | Train: 64.6% | Val: 35.1% | Loss V: 2.6510 | LR: 0.000800                          
Ep 008 | Train: 69.9% | Val: 52.6% | Loss V: 2.1804 | LR: 0.000800                          
Ep 009 | Train: 72.0% | Val: 58.5% | Loss V: 1.9654 | LR: 0.000800                          
Ep 010 | Train: 75.9% | Val: 50.5% | Loss V: 2.3197 | LR: 0.000800                          
Ep 011 | Train: 77.9% | Val: 52.4% | Loss V: 2.2111 | LR: 0.000800                          
Ep 012 | Train: 80.1% | Val: 53.8% | Loss V: 2.2466 | LR: 0.000800                          
Ep 013 | Train: 81.1% | Val: 64.3% | Loss V: 1.7933 | LR: 0.000800                          
Ep 014 | Train: 82.3% | Val: 58.5% | Loss V: 1.9293 | LR: 0.000800                          
Ep 015 | Train: 83.4% | Val: 61.8% | Loss V: 1.8365 | LR: 0.000800                          
Ep 016 | Train: 86.3% | Val: 53.0% | Loss V: 2.0928 | LR: 0.000800                          
Ep 017 | Train: 86.3% | Val: 69.2% | Loss V: 1.7087 | LR: 0.000800                          
Ep 018 | Train: 87.8% | Val: 74.5% | Loss V: 1.6128 | LR: 0.000800                          
Ep 019 | Train: 88.2% | Val: 60.8% | Loss V: 1.8934 | LR: 0.000800                          
Ep 020 | Train: 88.4% | Val: 65.7% | Loss V: 1.7828 | LR: 0.000800                          
Ep 021 | Train: 88.8% | Val: 60.4% | Loss V: 2.0293 | LR: 0.000800                          
Ep 022 | Train: 89.9% | Val: 59.8% | Loss V: 1.9440 | LR: 0.000800                          
Ep 023 | Train: 91.1% | Val: 69.8% | Loss V: 1.6581 | LR: 0.000800                          
Ep 024 | Train: 91.8% | Val: 80.5% | Loss V: 1.3644 | LR: 0.000800                          
Ep 025 | Train: 91.6% | Val: 66.7% | Loss V: 1.7570 | LR: 0.000800                          
Ep 026 | Train: 92.4% | Val: 81.7% | Loss V: 1.4400 | LR: 0.000800                          
Ep 027 | Train: 92.8% | Val: 79.5% | Loss V: 1.3911 | LR: 0.000800                          
Ep 028 | Train: 93.3% | Val: 69.2% | Loss V: 1.7165 | LR: 0.000800                          
Ep 029 | Train: 93.3% | Val: 63.2% | Loss V: 1.8791 | LR: 0.000800                          
Ep 030 | Train: 94.0% | Val: 75.8% | Loss V: 1.5267 | LR: 0.000400                          
Ep 031 | Train: 95.3% | Val: 78.9% | Loss V: 1.4415 | LR: 0.000400                          
Ep 032 | Train: 96.0% | Val: 80.1% | Loss V: 1.3739 | LR: 0.000400                          
Ep 033 | Train: 96.6% | Val: 77.8% | Loss V: 1.4351 | LR: 0.000400                          
Ep 034 | Train: 96.2% | Val: 76.2% | Loss V: 1.4492 | LR: 0.000400                          
Ep 035 | Train: 96.6% | Val: 80.5% | Loss V: 1.3941 | LR: 0.000400                          
Ep 036 | Train: 97.1% | Val: 80.9% | Loss V: 1.3880 | LR: 0.000200                          
Ep 037 | Train: 97.3% | Val: 80.5% | Loss V: 1.3934 | LR: 0.000200                          
Ep 038 | Train: 97.7% | Val: 81.3% | Loss V: 1.2907 | LR: 0.000200                          
Ep 039 | Train: 97.7% | Val: 80.9% | Loss V: 1.3387 | LR: 0.000200                          
Ep 040 | Train: 97.9% | Val: 79.7% | Loss V: 1.3719 | LR: 0.000200                          
Ep 041 | Train: 97.5% | Val: 81.9% | Loss V: 1.3465 | LR: 0.000200                          
Ep 042 | Train: 98.3% | Val: 81.7% | Loss V: 1.2915 | LR: 0.000200                          
Ep 043 | Train: 97.6% | Val: 80.7% | Loss V: 1.3680 | LR: 0.000200                          
Ep 044 | Train: 98.0% | Val: 83.0% | Loss V: 1.2855 | LR: 0.000200                          
Ep 045 | Train: 98.0% | Val: 82.8% | Loss V: 1.3053 | LR: 0.000200                          
Ep 046 | Train: 98.3% | Val: 80.3% | Loss V: 1.3910 | LR: 0.000200                          
Ep 047 | Train: 98.3% | Val: 84.2% | Loss V: 1.2607 | LR: 0.000200                          
Ep 048 | Train: 98.5% | Val: 84.2% | Loss V: 1.3129 | LR: 0.000200                          
Ep 049 | Train: 98.3% | Val: 81.9% | Loss V: 1.3132 | LR: 0.000200                          
Ep 050 | Train: 98.3% | Val: 83.4% | Loss V: 1.3008 | LR: 0.000200                          
Ep 051 | Train: 98.2% | Val: 80.5% | Loss V: 1.3694 | LR: 0.000200                          
Ep 052 | Train: 98.0% | Val: 78.2% | Loss V: 1.4114 | LR: 0.000200                          
Ep 053 | Train: 98.4% | Val: 81.3% | Loss V: 1.3207 | LR: 0.000100                          
Ep 054 | Train: 98.6% | Val: 81.9% | Loss V: 1.3031 | LR: 0.000100                          
Ep 055 | Train: 98.7% | Val: 80.3% | Loss V: 1.3562 | LR: 0.000100                          
Ep 056 | Train: 98.6% | Val: 79.1% | Loss V: 1.3676 | LR: 0.000100                          
Ep 057 | Train: 98.6% | Val: 82.1% | Loss V: 1.3097 | LR: 0.000100                          
Ep 058 | Train: 98.4% | Val: 80.3% | Loss V: 1.3528 | LR: 0.000100                          
Ep 059 | Train: 98.9% | Val: 81.3% | Loss V: 1.3354 | LR: 0.000050                          
Ep 060 | Train: 98.8% | Val: 81.9% | Loss V: 1.2912 | LR: 0.000050                          
Ep 061 | Train: 98.8% | Val: 81.7% | Loss V: 1.2979 | LR: 0.000050                          
Ep 062 | Train: 98.9% | Val: 81.1% | Loss V: 1.3205 | LR: 0.000050                          
Ep 063 | Train: 98.8% | Val: 81.1% | Loss V: 1.3241 | LR: 0.000050                          
Ep 064 | Train: 99.0% | Val: 81.5% | Loss V: 1.2785 | LR: 0.000050                          
Ep 065 | Train: 99.1% | Val: 78.4% | Loss V: 1.3516 | LR: 0.000025                          
Ep 066 | Train: 98.7% | Val: 80.7% | Loss V: 1.3258 | LR: 0.000025                          
Ep 067 | Train: 99.0% | Val: 79.9% | Loss V: 1.3199 | LR: 0.000025 


              precision    recall  f1-score   support

     id10270       0.94      1.00      0.97        16
     id10271       0.54      1.00      0.70         7
     id10272       0.71      1.00      0.83         5
     id10273       0.70      0.96      0.81        24
     id10274       1.00      1.00      1.00         5
     id10275       1.00      0.71      0.83         7
     id10276       1.00      0.58      0.73        19
     id10277       0.80      0.67      0.73         6
     id10278       0.88      0.79      0.83        19
     id10279       0.86      1.00      0.92         6
     id10280       1.00      0.67      0.80         6
     id10281       0.67      1.00      0.80         8
     id10282       1.00      1.00      1.00         8
     id10283       0.76      0.92      0.83        24
     id10284       1.00      0.78      0.88         9
     id10285       1.00      1.00      1.00        10
     id10286       1.00      0.93      0.97        15
     id10287       0.67      1.00      0.80         4
     id10288       0.75      0.75      0.75         4
     id10289       1.00      1.00      1.00         8
     id10290       0.92      0.86      0.89        14
     id10291       0.88      1.00      0.93         7
     id10292       0.87      1.00      0.93        27
     id10293       0.90      0.90      0.90        20
     id10294       0.87      0.93      0.90        14
     id10295       1.00      0.78      0.88         9
     id10296       1.00      0.70      0.82        10
     id10297       1.00      0.75      0.86         8
     id10298       0.68      1.00      0.81        13
     id10299       1.00      0.20      0.33         5
     id10300       0.96      0.87      0.92        31
     id10301       1.00      0.50      0.67         4
     id10302       1.00      0.53      0.69        17
     id10303       1.00      0.91      0.95        11
     id10304       1.00      0.50      0.67        16
     id10305       0.78      1.00      0.88        14
     id10306       1.00      1.00      1.00        19
     id10307       1.00      1.00      1.00        16
     id10308       0.86      1.00      0.92         6
     id10309       0.68      0.88      0.77        17

    accuracy                           0.86       488
   macro avg       0.89      0.85      0.85       488
weighted avg       0.89      0.86      0.86       488
