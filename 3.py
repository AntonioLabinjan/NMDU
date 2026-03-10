# add augmentacija
# eksperimentirat s layerima i veličinon modela

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
