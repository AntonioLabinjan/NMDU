import torch
import torch.nn as nn
import torch.nn.functional as F
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
BATCH_SIZE = 32 # Smanjeno zbog kompleksnije arhitekture
EPOCHS = 60
MAX_LR = 1e-3
WEIGHT_DECAY = 0.05

def get_run_folder(base_name="run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"): n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()
print(f"\n🔥 START: ResNet-SE Diagnostic Run in {RUN_DIR}")

# --- MODEL COMPONENTS (ResNet + Attention) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_f, out_f, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_f)
        self.conv2 = nn.Conv2d(out_f, out_f, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_f)
        self.se = SEBlock(out_f)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_f != out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, out_f, 1, stride, bias=False),
                nn.BatchNorm2d(out_f)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

class VoiceResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 4 faze s po 2 bloka (ukupno 8 konvolucijskih blokova)
        self.layer1 = self._make_layer(32, 32, stride=1)
        self.layer2 = self._make_layer(32, 64, stride=2)
        self.layer3 = self._make_layer(64, 128, stride=2)
        self.layer4 = self._make_layer(128, 256, stride=2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_f, out_f, stride):
        layers = [ResBlock(in_f, out_f, stride), ResBlock(out_f, out_f, 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
        self.freq_mask = torchaudio.transforms.FrequencyMasking(15) 
        self.time_mask = torchaudio.transforms.TimeMasking(30)

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        path, label = self.file_paths[idx], self.labels[idx]
        try:
            waveform, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
            
            if self.augment:
                waveform = waveform + 0.003 * torch.randn_like(waveform)

            mel_spec = self.db_transform(self.mel_transform(waveform))
            
            if self.augment:
                mel_spec = self.freq_mask(mel_spec)
                mel_spec = self.time_mask(mel_spec)

            mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear').squeeze(0)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)
            return mel_spec, label
        except:
            return torch.zeros((1, 128, 128)), label

# --- PREP DATA ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]
all_files, all_labels = [], []
for label, folder in enumerate(person_folders):
    for f in glob.glob(os.path.join(folder, "**/*.wav"), recursive=True):
        all_files.append(f); all_labels.append(label)

weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

X_train, X_temp, y_train, y_temp = train_test_split(all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_loader = DataLoader(VoiceDataset(X_train, y_train, augment=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VoiceDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(VoiceDataset(X_test, y_test), batch_size=BATCH_SIZE)

# --- SETUP ---
model = VoiceResNet(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)

history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_total = 0, 0, 0
    for bx, by in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        scheduler.step() # Unutar batch-a!
        
        t_loss += loss.item() * bx.size(0)
        t_corr += (out.argmax(1) == by).sum().item()
        t_total += bx.size(0)

    model.eval()
    v_loss, v_corr, v_total = 0, 0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx)
            v_loss += criterion(out, by).item() * bx.size(0)
            v_corr += (out.argmax(1) == by).sum().item()
            v_total += bx.size(0)

    history['t_loss'].append(t_loss/t_total); history['v_loss'].append(v_loss/v_total)
    history['t_acc'].append(100*t_corr/t_total); history['v_acc'].append(100*v_corr/v_total)
    
    print(f"Ep {epoch+1:03d} | T-Acc: {history['t_acc'][-1]:.1f}% | V-Acc: {history['v_acc'][-1]:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if history['v_acc'][-1] > best_acc:
        best_acc = history['v_acc'][-1]
        torch.save({'model': model.state_dict(), 'classes': class_names}, os.path.join(RUN_DIR, "best_model.pth"))

# --- DIAGNOSTICS ---
model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_model.pth"))['model'])
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for bx, by in test_loader:
        out = model(bx.to(DEVICE))
        y_pred.extend(out.argmax(1).cpu().numpy()); y_true.extend(by.numpy())

plt.figure(figsize=(15,5))
plt.subplot(1,2,1); plt.plot(history['t_loss'], label='Train'); plt.plot(history['v_loss'], label='Val'); plt.legend(); plt.title("Loss")
plt.subplot(1,2,2); plt.plot(history['t_acc'], label='Train'); plt.plot(history['v_acc'], label='Val'); plt.legend(); plt.title("Acc %")
plt.savefig(os.path.join(RUN_DIR, "curves.png"))

plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.savefig(os.path.join(RUN_DIR, "cm.png"))

print(f"✅ GOTOVO. Rezultati u {RUN_DIR}")
