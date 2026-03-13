import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 48
EPOCHS = 100
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 0.02
EARLY_STOP_PATIENCE = 20

def get_run_folder(base_name="run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"): n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()
print(f"\n🚀 START: Full Diagnostic Run in {RUN_DIR}")

# --- MODEL (VoiceNetDeep) ---
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
            conv_block(128, 256),
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

# --- OPTIMIZED DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        
        # Initialize transforms once for speed
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
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
            shift = int(waveform.shape[1] * 0.1)
            shift_val = np.random.randint(-shift, shift)
            waveform = torch.roll(waveform, shift_val, dims=1)
            waveform = waveform + 0.002 * torch.randn_like(waveform)

        mel_spec = self.mel_transform(waveform)
        mel_spec = self.db_transform(mel_spec)
        
        if self.augment:
            mel_spec = self.freq_mask(mel_spec)
            mel_spec = self.time_mask(mel_spec)
            
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)
        return mel_spec, label

# --- DATA PREPARATION ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]

all_files, all_labels = [], []
for label, folder in enumerate(person_folders):
    for f in glob.glob(os.path.join(folder, "**/*.wav"), recursive=True):
        all_files.append(f); all_labels.append(label)

# Calculate Class Weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

X_train_f, X_temp_f, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
X_val_f, X_test_f, y_val, y_test = train_test_split(X_temp_f, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(VoiceDataset(X_train_f, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(VoiceDataset(X_val_f, y_val, augment=False), batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(VoiceDataset(X_test_f, y_test, augment=False), batch_size=BATCH_SIZE)

# --- TRAINING SETUP ---
model = VoiceNetDeep(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
best_v_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False)
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
    print(f"Ep {epoch+1:03d} | Train Acc: {ta:.1f}% | Val Acc: {va:.1f}% | Val Loss: {vl:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if vl < best_v_loss:
        best_v_loss = vl
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= EARLY_STOP_PATIENCE:
        print("\n[!] Early stopping triggered.")
        break

# --- DIAGNOSTICS & RESULTS ---
print(f"\n📊 Generating reports in {RUN_DIR}...")

# 1. Training Curves
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['t_loss'], label='Train Loss', color='#1f77b4', lw=2)
plt.plot(history['v_loss'], label='Val Loss', color='#ff7f0e', lw=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['t_acc'], label='Train Acc', color='#2ca02c', lw=2)
plt.plot(history['v_acc'], label='Val Acc', color='#d62728', lw=2)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy %'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "training_metrics.png"))

# 2. Confusion Matrix & Classification Report
model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth")))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(DEVICE))
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(by.numpy())

with open(os.path.join(RUN_DIR, "detailed_report.txt"), "w") as f:
    f.write("--- CLASSIFICATION REPORT ---\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Speaker Classification')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))

print(f"✅ FINISHED. Check the {RUN_DIR} folder for your results.")


--- CLASSIFICATION REPORT ---
              precision    recall  f1-score   support

     id10270       0.94      1.00      0.97        16
     id10271       0.44      1.00      0.61         7
     id10272       0.83      1.00      0.91         5
     id10273       0.92      0.96      0.94        24
     id10274       0.62      1.00      0.77         5
     id10275       1.00      0.86      0.92         7
     id10276       1.00      0.47      0.64        19
     id10277       0.71      0.83      0.77         6
     id10278       1.00      0.74      0.85        19
     id10279       0.67      1.00      0.80         6
     id10280       1.00      0.83      0.91         6
     id10281       0.78      0.88      0.82         8
     id10282       1.00      1.00      1.00         8
     id10283       0.73      0.92      0.81        24
     id10284       0.80      0.89      0.84         9
     id10285       0.91      1.00      0.95        10
     id10286       1.00      1.00      1.00        15
     id10287       0.50      1.00      0.67         4
     id10288       0.80      1.00      0.89         4
     id10289       1.00      0.88      0.93         8
     id10290       0.92      0.86      0.89        14
     id10291       0.88      1.00      0.93         7
     id10292       0.93      0.96      0.95        27
     id10293       1.00      0.95      0.97        20
     id10294       0.78      1.00      0.88        14
     id10295       1.00      0.89      0.94         9
     id10296       0.90      0.90      0.90        10
     id10297       1.00      0.75      0.86         8
     id10298       0.54      1.00      0.70        13
     id10299       1.00      0.80      0.89         5
     id10300       0.96      0.81      0.88        31
     id10301       0.80      1.00      0.89         4
     id10302       0.88      0.41      0.56        17
     id10303       1.00      0.91      0.95        11
     id10304       1.00      0.25      0.40        16
     id10305       0.88      1.00      0.93        14
     id10306       0.94      0.89      0.92        19
     id10307       1.00      1.00      1.00        16
     id10308       1.00      1.00      1.00         6
     id10309       0.88      0.82      0.85        17

    accuracy                           0.86       488
   macro avg       0.87      0.89      0.86       488
weighted avg       0.90      0.86      0.86       488
