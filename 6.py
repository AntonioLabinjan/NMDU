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
EPOCHS = 80 # 80
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 15 

# --- LOGGING SETUP ---
def get_run_folder(base_name="run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"):
        n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()
print(f"\n🚀 START: Svi rezultati idu u folder: {RUN_DIR}")

# --- MODEL (VoiceNet Medium) ---
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
            sample_rate=SAMPLE_RATE, 
            n_mels=128, 
            n_fft=1024, 
            hop_length=512
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

# --- PRIPREMA PODATAKA ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]

config_data = {
    "run_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "lr": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "num_classes": len(class_names),
    "early_stop_patience": EARLY_STOP_PATIENCE
}
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(config_data, f, indent=4)

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
print(f"Uređaj za trening: {DEVICE}")
print("-" * 30)

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

    # Izračun metrika
    curr_train_loss = t_loss / t_total
    curr_val_loss = v_loss / v_total
    curr_train_acc = 100 * t_corr / t_total
    curr_val_acc = 100 * v_corr / v_total

    history['train_loss'].append(curr_train_loss)
    history['val_loss'].append(curr_val_loss)
    history['train_acc'].append(curr_train_acc)
    history['val_acc'].append(curr_val_acc)
    
    scheduler.step(curr_val_loss)
    
    # ISPRAVLJEN PRINT: Dodan Train Acc za bolji nadzor
    print(f"Epoch {epoch+1:03d} | Train Acc: {curr_train_acc:>5.1f}% | Val Acc: {curr_val_acc:>5.1f}% | Train Loss: {curr_train_loss:.3f} | Val Loss: {curr_val_loss:.3f}")

    if curr_val_loss < best_val_loss:
        best_val_loss = curr_val_loss
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n[!] EARLY STOPPING aktiviran na epohi {epoch+1}")
        break

# --- FINALNI IZVJEŠTAJI (Identično kao prije) ---
def save_reports():
    print("\nGeneriram finalne grafikone i izvještaje...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(RUN_DIR, "training_curves.png"))

    sample_spec, _ = next(iter(val_loader))
    plt.figure(figsize=(6, 4))
    #plt.imshow(sample_spec[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.imshow(sample_spec[0].squeeze().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Model Input Sample")
    plt.savefig(os.path.join(RUN_DIR, "input_sample.png"))

    model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth")))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(DEVICE))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(by.numpy())

    with open(os.path.join(RUN_DIR, "report.txt"), "w") as f:
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))
    plt.close('all')

save_reports()
print(f"✅ Sve završeno. Rezultati u: {RUN_DIR}")

Best Val Loss: 1.9585
------------------------------
              precision    recall  f1-score   support

     id10270       0.91      0.62      0.74        16
     id10271       0.40      0.29      0.33         7
     id10272       0.60      0.60      0.60         5
     id10273       0.28      0.88      0.43        24
     id10274       1.00      0.20      0.33         5
     id10275       0.50      0.14      0.22         7
     id10276       0.80      0.21      0.33        19
     id10277       0.75      0.50      0.60         6
     id10278       1.00      0.21      0.35        19
     id10279       1.00      0.17      0.29         6
     id10280       0.75      0.50      0.60         6
     id10281       0.00      0.00      0.00         8
     id10282       1.00      0.50      0.67         8
     id10283       0.28      0.67      0.40        24
     id10284       0.60      0.67      0.63         9
     id10285       0.69      0.90      0.78        10
     id10286       0.75      0.60      0.67        15
     id10287       0.33      0.25      0.29         4
     id10288       0.80      1.00      0.89         4
     id10289       0.50      0.38      0.43         8
     id10290       0.00      0.00      0.00        14
     id10291       0.42      0.71      0.53         7
     id10292       0.93      0.48      0.63        27
     id10293       1.00      0.55      0.71        20
     id10294       0.71      0.86      0.77        14
     id10295       0.50      0.22      0.31         9
     id10296       0.00      0.00      0.00        10
     id10297       1.00      0.12      0.22         8
     id10298       0.48      0.92      0.63        13
     id10299       0.00      0.00      0.00         5
     id10300       0.96      0.71      0.81        31
     id10301       0.00      0.00      0.00         4
     id10302       0.75      0.35      0.48        17
     id10303       0.53      0.82      0.64        11
     id10304       0.00      0.00      0.00        16
     id10305       0.40      0.57      0.47        14
     id10306       0.49      0.89      0.63        19
     id10307       0.67      0.62      0.65        16
     id10308       1.00      0.50      0.67         6
     id10309       0.26      0.76      0.39        17

    accuracy                           0.51       488
   macro avg       0.58      0.46      0.45       488
weighted avg       0.60      0.51      0.49       488
