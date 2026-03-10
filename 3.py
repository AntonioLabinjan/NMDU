### 📋 TODO: Voice Classification Optimization

* **1. 🔊 Augmentacija podataka (SpecAugment)**
* [ ] Implementirati `torchaudio.transforms.FrequencyMasking(freq_mask_param=15)`
* [ ] Implementirati `torchaudio.transforms.TimeMasking(time_mask_param=35)`
* [ ] Dodati Gaussian White Noise na waveform (`0.005 * torch.randn_like`)
* [ ] Nasumični Time Stretch (0.8x - 1.2x)


* **2. ✂️ Arhitektura (Smanjivanje modela)**
* [ ] Reducirati broj filtera: `16 -> 32 -> 64` (umjesto `32 -> 64 -> 128 -> 256`)
* [ ] Izbaciti zadnji `conv_block` (ostati na 3 sloja)
* [ ] Povećati `Dropout` u klasifikatoru na `0.6`
* [ ] Ubaciti `nn.Dropout2d(0.2)` nakon svakog konvolucijskog bloka


* **3. 🧪 Trening i Regularizacija**
* [ ] Uvesti `nn.CrossEntropyLoss(label_smoothing=0.1)`
* [ ] Povećati `weight_decay` u `AdamW` na `0.05` ili `0.1`
* [ ] Implementirati `torch.optim.lr_scheduler.ReduceLROnPlateau`
* [ ] Smanjiti početni `LEARNING_RATE` na `0.0001`


* **4. 📊 Kvaliteta podataka**
* [ ] Podići `SAMPLES_PER_PERSON` na minimalno 30 uzoraka
* [ ] Provjeriti balans klasa (ujednačen broj snimki po govorniku)
* [ ] Normalizirati glasnoću svih snimki (Peak normalization)

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005

# --- MODEL (ResNet-ish) ---
class VoiceResNetish(nn.Module):
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
            conv_block(1,32), conv_block(32,64),
            conv_block(64,128), conv_block(128,256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

# --- PREPROCESS ---
def load_and_preprocess(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    mel_spec = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_mels=64)(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec = torch.nn.functional.interpolate(
        mel_spec.unsqueeze(0), size=(128,128), mode='bilinear', align_corners=False
    ).squeeze(0)
    return (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)

# --- PREPARE DATA ---
print(f"--- START: {datetime.now().strftime('%H:%M:%S')} ---")
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
class_names = [os.path.basename(f) for f in person_folders]

# --- AUTOMATSKI SAMPLES_PER_PERSON ---
samples_per_person = min(len(glob.glob(os.path.join(f, "**/*.wav"), recursive=True)) for f in person_folders)
print(f"[DATA] Broj uzoraka po govorniku: {samples_per_person}")

X_all, y_all = [], []
for label, folder in enumerate(person_folders):
    files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)[:samples_per_person]
    for f in files:
        X_all.append(load_and_preprocess(f))
        y_all.append(label)

X_all = torch.stack(X_all)
y_all = torch.tensor(y_all)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"[DATA] Trening: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

model = VoiceResNetish(len(class_names)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')

print(f"[INIT] Model spreman na {DEVICE}. Krećem s treningom...\n")

# --- TRAINING LOOP ---
start_train_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        correct += (outputs.argmax(1) == batch_y).sum().item()
        total += batch_x.size(0)
    train_loss = running_loss / total
    train_acc = correct / total * 100

    # --- VAL ---
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * batch_x.size(0)
            correct += (outputs.argmax(1) == batch_y).sum().item()
            total += batch_x.size(0)
    val_loss = running_loss / total
    val_acc = correct / total * 100

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_voice_model.pth")

# --- FINAL EVAL ---
print(f"\n[DONE] Trening gotov. Ukupno vrijeme: {str(timedelta(seconds=int(time.time() - start_train_time)))}")
model.load_state_dict(torch.load("best_voice_model.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        outputs = model(batch_x)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(batch_y.numpy())

print("\n" + "="*40)
print(" FINALNI IZVJEŠTAJ (TEST SET) ")
print("="*40)
print(classification_report(y_true, y_pred, target_names=class_names))


(venv_voice311) antonio@antonio-IdeaPad-Slim-3-15IAH8:~/Desktop/NMDU$ python3 neural_network.py
--- START: 14:02:02 ---
[DATA] Broj uzoraka po govorniku: 48
[DATA] Trening: 1536 | Val: 192 | Test: 192
[INIT] Model spreman na cpu. Krećem s treningom...

Epoch 001/100 | Train Loss: 3.6659, Train Acc: 5.01% | Val Loss: 3.5613, Val Acc: 5.73%
Epoch 002/100 | Train Loss: 3.4805, Train Acc: 8.27% | Val Loss: 3.5501, Val Acc: 5.21%
Epoch 003/100 | Train Loss: 3.2218, Train Acc: 12.83% | Val Loss: 3.1868, Val Acc: 9.90%
Epoch 004/100 | Train Loss: 2.9738, Train Acc: 17.58% | Val Loss: 2.8687, Val Acc: 20.31%
Epoch 005/100 | Train Loss: 2.7631, Train Acc: 22.01% | Val Loss: 2.6471, Val Acc: 31.25%
Epoch 006/100 | Train Loss: 2.6090, Train Acc: 23.70% | Val Loss: 2.9586, Val Acc: 21.35%
Epoch 007/100 | Train Loss: 2.4438, Train Acc: 30.47% | Val Loss: 2.6728, Val Acc: 26.04%
Epoch 008/100 | Train Loss: 2.3330, Train Acc: 33.07% | Val Loss: 2.6854, Val Acc: 22.92%
Epoch 009/100 | Train Loss: 2.2193, Train Acc: 34.18% | Val Loss: 2.8467, Val Acc: 19.27%
Epoch 010/100 | Train Loss: 2.0943, Train Acc: 39.97% | Val Loss: 2.2491, Val Acc: 29.69%
Epoch 011/100 | Train Loss: 1.9567, Train Acc: 42.84% | Val Loss: 2.5260, Val Acc: 29.69%
Epoch 012/100 | Train Loss: 1.8855, Train Acc: 45.64% | Val Loss: 1.8818, Val Acc: 46.88%
Epoch 013/100 | Train Loss: 1.7831, Train Acc: 48.11% | Val Loss: 2.6891, Val Acc: 31.25%
Epoch 014/100 | Train Loss: 1.7009, Train Acc: 51.17% | Val Loss: 2.3952, Val Acc: 29.17%
Epoch 015/100 | Train Loss: 1.6640, Train Acc: 51.04% | Val Loss: 2.4542, Val Acc: 30.73%
Epoch 016/100 | Train Loss: 1.5584, Train Acc: 55.27% | Val Loss: 1.9918, Val Acc: 44.79%
Epoch 017/100 | Train Loss: 1.4669, Train Acc: 57.55% | Val Loss: 2.6681, Val Acc: 30.73%
Epoch 018/100 | Train Loss: 1.4106, Train Acc: 59.44% | Val Loss: 2.2174, Val Acc: 31.25%
Epoch 019/100 | Train Loss: 1.3555, Train Acc: 61.33% | Val Loss: 3.6039, Val Acc: 13.54%
Epoch 020/100 | Train Loss: 1.2751, Train Acc: 63.15% | Val Loss: 3.0857, Val Acc: 24.48%
Epoch 021/100 | Train Loss: 1.2449, Train Acc: 63.48% | Val Loss: 2.0832, Val Acc: 41.15%
Epoch 022/100 | Train Loss: 1.1746, Train Acc: 65.82% | Val Loss: 2.5233, Val Acc: 35.42%
Epoch 023/100 | Train Loss: 1.1463, Train Acc: 65.49% | Val Loss: 3.1520, Val Acc: 22.40%
Epoch 024/100 | Train Loss: 1.0335, Train Acc: 69.86% | Val Loss: 2.1974, Val Acc: 36.98%
Epoch 025/100 | Train Loss: 0.9985, Train Acc: 69.92% | Val Loss: 2.6776, Val Acc: 35.94%
Epoch 026/100 | Train Loss: 0.9610, Train Acc: 71.81% | Val Loss: 2.2522, Val Acc: 38.54%
Epoch 027/100 | Train Loss: 0.9006, Train Acc: 72.79% | Val Loss: 3.2779, Val Acc: 28.12%
Epoch 028/100 | Train Loss: 0.8675, Train Acc: 74.48% | Val Loss: 2.7143, Val Acc: 34.90%
Epoch 029/100 | Train Loss: 0.8893, Train Acc: 72.72% | Val Loss: 2.8509, Val Acc: 27.08%
Epoch 030/100 | Train Loss: 0.8338, Train Acc: 75.72% | Val Loss: 2.4435, Val Acc: 35.42%
Epoch 031/100 | Train Loss: 0.7991, Train Acc: 76.37% | Val Loss: 1.6660, Val Acc: 49.48%
Epoch 032/100 | Train Loss: 0.7403, Train Acc: 78.26% | Val Loss: 1.5944, Val Acc: 57.29%
Epoch 033/100 | Train Loss: 0.7075, Train Acc: 79.30% | Val Loss: 2.0916, Val Acc: 46.35%
Epoch 034/100 | Train Loss: 0.6812, Train Acc: 80.79% | Val Loss: 3.4731, Val Acc: 25.52%
Epoch 035/100 | Train Loss: 0.6145, Train Acc: 81.97% | Val Loss: 1.6778, Val Acc: 54.69%
Epoch 036/100 | Train Loss: 0.6494, Train Acc: 80.86% | Val Loss: 3.0378, Val Acc: 30.73%
Epoch 037/100 | Train Loss: 0.6672, Train Acc: 79.36% | Val Loss: 1.7405, Val Acc: 51.04%
Epoch 038/100 | Train Loss: 0.5943, Train Acc: 81.64% | Val Loss: 1.4663, Val Acc: 60.42%
Epoch 039/100 | Train Loss: 0.5882, Train Acc: 80.99% | Val Loss: 2.4829, Val Acc: 39.58%
Epoch 040/100 | Train Loss: 0.5400, Train Acc: 83.27% | Val Loss: 3.8326, Val Acc: 26.56%
Epoch 041/100 | Train Loss: 0.5561, Train Acc: 82.42% | Val Loss: 2.1638, Val Acc: 44.79%
Epoch 042/100 | Train Loss: 0.4894, Train Acc: 84.70% | Val Loss: 2.7151, Val Acc: 39.58%
Epoch 043/100 | Train Loss: 0.5033, Train Acc: 84.90% | Val Loss: 2.2732, Val Acc: 44.27%
Epoch 044/100 | Train Loss: 0.4804, Train Acc: 85.42% | Val Loss: 1.7574, Val Acc: 52.08%
Epoch 045/100 | Train Loss: 0.4879, Train Acc: 84.57% | Val Loss: 3.2274, Val Acc: 30.73%
Epoch 046/100 | Train Loss: 0.4613, Train Acc: 86.46% | Val Loss: 3.3308, Val Acc: 28.65%
Epoch 047/100 | Train Loss: 0.4088, Train Acc: 86.78% | Val Loss: 1.6593, Val Acc: 55.73%
Epoch 048/100 | Train Loss: 0.4178, Train Acc: 88.28% | Val Loss: 3.2553, Val Acc: 31.25%
Epoch 049/100 | Train Loss: 0.4523, Train Acc: 85.48% | Val Loss: 2.2713, Val Acc: 50.00%
Epoch 050/100 | Train Loss: 0.4155, Train Acc: 87.83% | Val Loss: 3.3785, Val Acc: 28.12%
Epoch 051/100 | Train Loss: 0.3830, Train Acc: 88.28% | Val Loss: 1.6318, Val Acc: 57.29%
Epoch 052/100 | Train Loss: 0.3435, Train Acc: 90.49% | Val Loss: 4.9410, Val Acc: 24.48%
Epoch 053/100 | Train Loss: 0.3813, Train Acc: 88.61% | Val Loss: 3.7366, Val Acc: 30.21%
Epoch 054/100 | Train Loss: 0.3663, Train Acc: 89.00% | Val Loss: 2.2236, Val Acc: 48.96%
Epoch 055/100 | Train Loss: 0.3302, Train Acc: 89.52% | Val Loss: 3.5065, Val Acc: 29.17%
Epoch 056/100 | Train Loss: 0.3459, Train Acc: 89.13% | Val Loss: 3.2272, Val Acc: 33.33%
Epoch 057/100 | Train Loss: 0.3605, Train Acc: 89.00% | Val Loss: 5.7043, Val Acc: 18.75%
Epoch 058/100 | Train Loss: 0.3163, Train Acc: 90.69% | Val Loss: 1.7959, Val Acc: 56.77%
Epoch 059/100 | Train Loss: 0.3309, Train Acc: 89.58% | Val Loss: 4.8748, Val Acc: 21.35%
Epoch 060/100 | Train Loss: 0.2937, Train Acc: 91.21% | Val Loss: 2.8664, Val Acc: 39.58%
Epoch 061/100 | Train Loss: 0.3078, Train Acc: 89.84% | Val Loss: 2.8816, Val Acc: 44.79%
Epoch 062/100 | Train Loss: 0.2898, Train Acc: 91.08% | Val Loss: 1.6843, Val Acc: 57.81%
Epoch 063/100 | Train Loss: 0.2626, Train Acc: 91.41% | Val Loss: 2.8147, Val Acc: 40.62%
Epoch 064/100 | Train Loss: 0.2548, Train Acc: 92.25% | Val Loss: 2.3946, Val Acc: 48.96%
Epoch 065/100 | Train Loss: 0.2596, Train Acc: 92.25% | Val Loss: 2.5849, Val Acc: 47.40%
Epoch 066/100 | Train Loss: 0.2285, Train Acc: 93.16% | Val Loss: 1.7125, Val Acc: 57.29%
Epoch 067/100 | Train Loss: 0.2142, Train Acc: 93.95% | Val Loss: 3.0328, Val Acc: 36.98%
Epoch 068/100 | Train Loss: 0.2221, Train Acc: 93.55% | Val Loss: 2.6577, Val Acc: 41.15%
Epoch 069/100 | Train Loss: 0.2592, Train Acc: 93.03% | Val Loss: 6.8347, Val Acc: 15.62%
Epoch 070/100 | Train Loss: 0.2771, Train Acc: 91.47% | Val Loss: 4.5091, Val Acc: 27.08%
Epoch 071/100 | Train Loss: 0.2270, Train Acc: 92.84% | Val Loss: 2.0177, Val Acc: 55.21%
Epoch 072/100 | Train Loss: 0.2034, Train Acc: 93.68% | Val Loss: 2.8661, Val Acc: 47.40%
Epoch 073/100 | Train Loss: 0.2406, Train Acc: 92.38% | Val Loss: 2.0195, Val Acc: 57.29%
Epoch 074/100 | Train Loss: 0.2378, Train Acc: 92.19% | Val Loss: 2.5902, Val Acc: 47.40%
Epoch 075/100 | Train Loss: 0.2119, Train Acc: 93.88% | Val Loss: 1.5751, Val Acc: 64.58%
Epoch 076/100 | Train Loss: 0.2206, Train Acc: 93.10% | Val Loss: 1.9756, Val Acc: 52.08%
Epoch 077/100 | Train Loss: 0.1988, Train Acc: 93.88% | Val Loss: 1.6508, Val Acc: 60.42%
Epoch 078/100 | Train Loss: 0.2307, Train Acc: 92.32% | Val Loss: 2.0182, Val Acc: 53.12%
Epoch 079/100 | Train Loss: 0.2006, Train Acc: 94.21% | Val Loss: 4.8818, Val Acc: 31.25%
Epoch 080/100 | Train Loss: 0.2051, Train Acc: 92.38% | Val Loss: 1.9708, Val Acc: 54.17%
Epoch 081/100 | Train Loss: 0.1718, Train Acc: 95.05% | Val Loss: 3.2311, Val Acc: 45.83%
Epoch 082/100 | Train Loss: 0.2307, Train Acc: 93.82% | Val Loss: 2.8951, Val Acc: 43.75%
Epoch 083/100 | Train Loss: 0.2015, Train Acc: 93.10% | Val Loss: 2.7059, Val Acc: 40.10%
Epoch 084/100 | Train Loss: 0.1751, Train Acc: 94.27% | Val Loss: 2.2801, Val Acc: 52.08%
Epoch 085/100 | Train Loss: 0.2071, Train Acc: 93.62% | Val Loss: 4.3300, Val Acc: 27.60%
Epoch 086/100 | Train Loss: 0.1891, Train Acc: 93.88% | Val Loss: 4.8009, Val Acc: 30.21%
Epoch 087/100 | Train Loss: 0.1960, Train Acc: 94.47% | Val Loss: 4.8616, Val Acc: 32.29%
Epoch 088/100 | Train Loss: 0.2053, Train Acc: 93.49% | Val Loss: 2.4637, Val Acc: 52.60%
Epoch 089/100 | Train Loss: 0.2057, Train Acc: 93.95% | Val Loss: 2.7083, Val Acc: 45.31%
Epoch 090/100 | Train Loss: 0.1706, Train Acc: 95.12% | Val Loss: 5.2937, Val Acc: 26.04%
Epoch 091/100 | Train Loss: 0.1558, Train Acc: 95.64% | Val Loss: 2.3268, Val Acc: 53.65%
Epoch 092/100 | Train Loss: 0.1751, Train Acc: 94.86% | Val Loss: 2.1301, Val Acc: 57.81%
Epoch 093/100 | Train Loss: 0.1789, Train Acc: 94.01% | Val Loss: 2.9869, Val Acc: 45.83%
Epoch 094/100 | Train Loss: 0.1651, Train Acc: 94.40% | Val Loss: 2.1931, Val Acc: 57.81%
Epoch 095/100 | Train Loss: 0.1545, Train Acc: 94.66% | Val Loss: 3.7287, Val Acc: 40.62%
Epoch 096/100 | Train Loss: 0.1392, Train Acc: 95.31% | Val Loss: 2.6379, Val Acc: 47.40%
Epoch 097/100 | Train Loss: 0.1322, Train Acc: 95.77% | Val Loss: 1.9340, Val Acc: 56.25%
Epoch 098/100 | Train Loss: 0.1266, Train Acc: 95.96% | Val Loss: 1.9221, Val Acc: 59.90%
Epoch 099/100 | Train Loss: 0.1847, Train Acc: 93.68% | Val Loss: 6.4053, Val Acc: 26.56%
Epoch 100/100 | Train Loss: 0.1497, Train Acc: 95.38% | Val Loss: 2.7816, Val Acc: 43.75%

[DONE] Trening gotov. Ukupno vrijeme: 0:50:11

========================================
 FINALNI IZVJEŠTAJ (TEST SET) 
========================================
/home/antonio/Desktop/NMDU/venv_voice311/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/home/antonio/Desktop/NMDU/venv_voice311/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
/home/antonio/Desktop/NMDU/venv_voice311/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1833: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

     id10270       1.00      0.80      0.89         5
     id10271       0.40      0.80      0.53         5
     id10272       0.31      0.80      0.44         5
     id10273       1.00      0.20      0.33         5
     id10274       0.00      0.00      0.00         4
     id10275       0.83      1.00      0.91         5
     id10276       0.43      0.75      0.55         4
     id10277       0.50      0.60      0.55         5
     id10278       0.00      0.00      0.00         5
     id10279       1.00      0.20      0.33         5
     id10280       1.00      0.20      0.33         5
     id10281       1.00      0.40      0.57         5
     id10282       1.00      0.60      0.75         5
     id10283       1.00      0.75      0.86         4
     id10284       1.00      0.80      0.89         5
     id10285       0.29      0.40      0.33         5
     id10286       0.80      0.80      0.80         5
     id10287       0.62      1.00      0.77         5
     id10288       0.50      0.20      0.29         5
     id10289       0.00      0.00      0.00         4
     id10290       1.00      0.60      0.75         5
     id10291       0.38      1.00      0.56         5
     id10292       1.00      0.80      0.89         5
     id10293       1.00      0.40      0.57         5
     id10294       1.00      1.00      1.00         5
     id10295       1.00      0.25      0.40         4
     id10296       0.67      0.40      0.50         5
     id10297       0.12      0.75      0.21         4
     id10298       0.27      0.60      0.38         5
     id10299       0.80      0.80      0.80         5
     id10300       0.80      0.80      0.80         5
     id10301       1.00      0.80      0.89         5
     id10302       0.67      0.40      0.50         5
     id10303       1.00      0.40      0.57         5
     id10304       0.50      0.50      0.50         4
     id10305       0.50      0.20      0.29         5
     id10306       1.00      0.60      0.75         5
     id10307       0.00      0.00      0.00         4
     id10308       1.00      0.80      0.89         5
     id10309       0.36      0.80      0.50         5

    accuracy                           0.56       192
   macro avg       0.67      0.55      0.55       192
weighted avg       0.68      0.56      0.56       192

