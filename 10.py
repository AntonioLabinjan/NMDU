# kinda shit
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
BATCH_SIZE = 32  # Slightly smaller for better generalization
EPOCHS = 120     # More epochs for potential grokking
MAX_LR = 1e-3
WEIGHT_DECAY = 0.05 # Higher weight decay for better regularization

def get_run_folder(base_name="run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"): n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()
print(f"\n🚀 START: Residual VoiceNet Training in {RUN_DIR}")

# --- MODEL COMPONENTS ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_f, out_f, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_f)
        self.conv2 = nn.Conv2d(out_f, out_f, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_f)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_f)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_f != out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, out_f, 1, stride, bias=False),
                nn.BatchNorm2d(out_f)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

class VoiceResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_planes = 32
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, planes, num_blocks):
        strides = [2] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)

# --- DATASET ---
class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(30) # Increased mask
        self.time_mask = torchaudio.transforms.TimeMasking(40) # Increased mask

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
            
        if self.augment:
            shift = int(waveform.shape[1] * 0.15) # Increased shift
            shift_val = np.random.randint(-shift, shift)
            waveform = torch.roll(waveform, shift_val, dims=1)
            waveform = waveform + 0.003 * torch.randn_like(waveform)

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

# --- PREPARATION ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]
all_files, all_labels = [], []
for label, folder in enumerate(person_folders):
    for f in glob.glob(os.path.join(folder, "**/*.wav"), recursive=True):
        all_files.append(f); all_labels.append(label)

weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

X_train_f, X_temp_f, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
X_val_f, X_test_f, y_val, y_test = train_test_split(X_temp_f, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(VoiceDataset(X_train_f, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(VoiceDataset(X_val_f, y_val, augment=False), batch_size=BATCH_SIZE, num_workers=4)
test_loader = DataLoader(VoiceDataset(X_test_f, y_test, augment=False), batch_size=BATCH_SIZE)

# --- TRAINING SETUP ---
model = VoiceResNet(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)

# OneCycleLR helps the model "grok" by finding better minima
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, 
                                          steps_per_epoch=len(train_loader), epochs=EPOCHS)

history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
best_v_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False)
    for bx, by in pbar:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad(); out = model(bx); loss = criterion(out, by)
        loss.backward(); optimizer.step(); scheduler.step()
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
    
    print(f"Ep {epoch+1:03d} | Train Acc: {ta:.1f}% | Val Acc: {va:.1f}% | Val Loss: {vl:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if vl < best_v_loss:
        best_v_loss = vl
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pth"))

# --- DIAGNOSTICS & RESULTS ---
print(f"\n📊 Generating reports in {RUN_DIR}...")
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['t_loss'], label='Train Loss'); plt.plot(history['v_loss'], label='Val Loss')
plt.title('Loss Curves'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['t_acc'], label='Train Acc'); plt.plot(history['v_acc'], label='Val Acc')
plt.title('Accuracy Curves'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(RUN_DIR, "training_metrics.png"))

model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth")))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(DEVICE))
        y_pred.extend(out.argmax(1).cpu().numpy()); y_true.extend(by.numpy())

with open(os.path.join(RUN_DIR, "detailed_report.txt"), "w") as f:
    f.write(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Final Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))

print(f"✅ FINISHED. The Residual Model is ready.")


              precision    recall  f1-score   support

     id10270       0.70      0.88      0.78        16
     id10271       0.30      1.00      0.47         7
     id10272       0.80      0.80      0.80         5
     id10273       0.75      0.75      0.75        24
     id10274       0.38      1.00      0.56         5
     id10275       1.00      0.57      0.73         7
     id10276       1.00      0.37      0.54        19
     id10277       0.19      0.67      0.30         6
     id10278       1.00      0.05      0.10        19
     id10279       1.00      1.00      1.00         6
     id10280       1.00      0.67      0.80         6
     id10281       0.80      1.00      0.89         8
     id10282       0.88      0.88      0.88         8
     id10283       0.59      0.79      0.68        24
     id10284       0.80      0.44      0.57         9
     id10285       0.59      1.00      0.74        10
     id10286       1.00      0.87      0.93        15
     id10287       0.33      1.00      0.50         4
     id10288       0.80      1.00      0.89         4
     id10289       0.78      0.88      0.82         8
     id10290       0.91      0.71      0.80        14
     id10291       0.88      1.00      0.93         7
     id10292       0.60      0.93      0.72        27
     id10293       0.86      0.60      0.71        20
     id10294       1.00      0.64      0.78        14
     id10295       0.35      1.00      0.51         9
     id10296       0.80      0.80      0.80        10
     id10297       0.67      0.75      0.71         8
     id10298       1.00      0.54      0.70        13
     id10299       0.83      1.00      0.91         5
     id10300       0.94      0.48      0.64        31
     id10301       0.50      0.50      0.50         4
     id10302       0.89      0.47      0.62        17
     id10303       1.00      0.82      0.90        11
     id10304       1.00      0.38      0.55        16
     id10305       1.00      0.29      0.44        14
     id10306       0.94      0.89      0.92        19
     id10307       0.94      0.94      0.94        16
     id10308       1.00      0.50      0.67         6
     id10309       0.68      0.88      0.77        17

    accuracy                           0.70       488
   macro avg       0.79      0.74      0.71       488
weighted avg       0.82      0.70      0.70       488
