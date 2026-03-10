import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import random
import time
from datetime import datetime, timedelta

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_PERSON = 5 
EPOCHS = 50 
LEARNING_RATE = 0.001

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
            conv_block(1, 32),   
            conv_block(32, 64),  
            conv_block(64, 128), 
            conv_block(128, 256) 
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- UTILS ---
def load_and_preprocess(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    
    mel_spec = torchaudio.transforms.MelSpectrogram(
        SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=256
    )(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    mel_spec = torch.nn.functional.interpolate(
        mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    mu, std = mel_spec.mean(), mel_spec.std()
    return (mel_spec - mu) / (std + 1e-7)

# --- INITIALIZATION ---
print(f"{'='*50}\n[INFO] Sustav za klasifikaciju glasova\n{'='*50}")
print(f"[*] Device: {DEVICE}")
print(f"[*] Vrijeme početka: {datetime.now().strftime('%H:%M:%S')}")

person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
NUM_CLASSES = len(person_folders)

X_list, y_list = [], []
start_load = time.time()

for label, folder in enumerate(person_folders):
    audio_files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)[:SAMPLES_PER_PERSON]
    for f in audio_files:
        X_list.append(load_and_preprocess(f))
        y_list.append(label)

X = torch.stack(X_list).to(DEVICE)
y = torch.tensor(y_list).to(DEVICE)

load_duration = time.time() - start_load
print(f"[*] Dataset učitan: {len(X)} uzoraka | Vrijeme: {load_duration:.2f}s")
print(f"[*] Broj klasa (osoba): {NUM_CLASSES}")

model = VoiceResNetish(num_classes=NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"[*] Broj parametara modela: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# --- TRAINING WITH ETA ---
print(f"\n{'='*20} TRENING {'='*20}")
start_train = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    # Metrike
    with torch.no_grad():
        pred = outputs.argmax(1)
        acc = (pred == y).float().mean() * 100
        
    # Kalkulacija vremena i ETA
    epoch_duration = time.time() - epoch_start_time
    elapsed_time = time.time() - start_train
    avg_time_per_epoch = elapsed_time / (epoch + 1)
    remaining_epochs = EPOCHS - (epoch + 1)
    eta_seconds = remaining_epochs * avg_time_per_epoch
    eta_str = str(timedelta(seconds=int(eta_seconds)))

    # Robustni ispis progresa
    progress_bar = f"[{'#' * int((epoch+1)/(EPOCHS/20))}{'-' * (20 - int((epoch+1)/(EPOCHS/20)))}]"
    print(f"Epoch {epoch+1:03d}/{EPOCHS} {progress_bar} | Loss: {loss.item():.4f} | Acc: {acc:.2f}% | ETA: {eta_str} | {1/epoch_duration:.1f} it/s", end='\r')

    if (epoch + 1) == EPOCHS: print() # Novi red na kraju

total_train_time = str(timedelta(seconds=int(time.time() - start_train)))
print(f"{'='*50}\n[DONE] Trening završen u: {total_train_time}\n{'='*50}")

# --- TEST RANDOM SAMPLE ---
model.eval()
with torch.no_grad():
    test_idx = random.randint(0, len(X) - 1)
    sample_input = X[test_idx].unsqueeze(0)
    actual_label = y[test_idx].item()
    
    start_inf = time.time()
    output = model(sample_input)
    inference_time = (time.time() - start_inf) * 1000
    
    pred_label = output.argmax(1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_label].item() * 100

    print(f"\n[INFERENCE TEST]")
    print(f"ID uzorka: {test_idx}")
    print(f"Stvarno:  {os.path.basename(person_folders[actual_label])}")
    print(f"Predviđeno: {os.path.basename(person_folders[pred_label])} ({confidence:.2f}% confidence)")
    print(f"Vrijeme inferencije: {inference_time:.2f}ms")
'''
==================================================
[INFO] Sustav za klasifikaciju glasova
==================================================
[*] Device: cpu
[*] Vrijeme početka: 13:29:56
[*] Dataset učitan: 200 uzoraka | Vrijeme: 2.76s
[*] Broj klasa (osoba): 40
[*] Broj parametara modela: 426,856

==================== TRENING ====================
Epoch 001/50 [--------------------] | Loss: 3.7164 | Acc: 3.00% | ETA: 0:01:55 |Epoch 002/50 [--------------------] | Loss: 3.6352 | Acc: 5.00% | ETA: 0:02:17 |Epoch 003/50 [#-------------------] | Loss: 3.5915 | Acc: 4.00% | ETA: 0:02:30 |Epoch 004/50 [#-------------------] | Loss: 3.5493 | Acc: 8.00% | ETA: 0:02:32 |Epoch 005/50 [##------------------] | Loss: 3.5116 | Acc: 7.00% | ETA: 0:02:36 |Epoch 006/50 [##------------------] | Loss: 3.4721 | Acc: 10.00% | ETA: 0:02:36 Epoch 007/50 [##------------------] | Loss: 3.4220 | Acc: 13.00% | ETA: 0:02:37 Epoch 008/50 [###-----------------] | Loss: 3.3783 | Acc: 10.00% | ETA: 0:02:36 Epoch 009/50 [###-----------------] | Loss: 3.3416 | Acc: 14.00% | ETA: 0:02:33 Epoch 010/50 [####----------------] | Loss: 3.2938 | Acc: 10.50% | ETA: 0:02:30 Epoch 011/50 [####----------------] | Loss: 3.2347 | Acc: 13.00% | ETA: 0:02:27 Epoch 012/50 [####----------------] | Loss: 3.1880 | Acc: 16.00% | ETA: 0:02:24 Epoch 013/50 [#####---------------] | Loss: 3.1361 | Acc: 15.50% | ETA: 0:02:20 Epoch 014/50 [#####---------------] | Loss: 3.0712 | Acc: 22.50% | ETA: 0:02:17 Epoch 015/50 [######--------------] | Loss: 3.0025 | Acc: 23.00% | ETA: 0:02:13 Epoch 016/50 [######--------------] | Loss: 2.9608 | Acc: 26.00% | ETA: 0:02:10 Epoch 017/50 [######--------------] | Loss: 2.9164 | Acc: 27.50% | ETA: 0:02:06 Epoch 018/50 [#######-------------] | Loss: 2.8617 | Acc: 30.00% | ETA: 0:02:02 Epoch 019/50 [#######-------------] | Loss: 2.7815 | Acc: 35.00% | ETA: 0:01:58 Epoch 020/50 [########------------] | Loss: 2.7449 | Acc: 36.00% | ETA: 0:01:54 Epoch 021/50 [########------------] | Loss: 2.6491 | Acc: 43.50% | ETA: 0:01:50 Epoch 022/50 [########------------] | Loss: 2.5787 | Acc: 43.50% | ETA: 0:01:46 Epoch 023/50 [#########-----------] | Loss: 2.5168 | Acc: 48.00% | ETA: 0:01:42 Epoch 024/50 [#########-----------] | Loss: 2.4545 | Acc: 51.50% | ETA: 0:01:38 Epoch 025/50 [##########----------] | Loss: 2.3605 | Acc: 52.00% | ETA: 0:01:34 Epoch 026/50 [##########----------] | Loss: 2.3074 | Acc: 61.50% | ETA: 0:01:30 Epoch 027/50 [##########----------] | Loss: 2.2540 | Acc: 60.00% | ETA: 0:01:27 Epoch 028/50 [###########---------] | Loss: 2.1551 | Acc: 64.50% | ETA: 0:01:23 Epoch 029/50 [###########---------] | Loss: 2.0913 | Acc: 65.50% | ETA: 0:01:19 Epoch 030/50 [############--------] | Loss: 2.0386 | Acc: 69.50% | ETA: 0:01:15 Epoch 031/50 [############--------] | Loss: 1.9374 | Acc: 72.00% | ETA: 0:01:11 Epoch 032/50 [############--------] | Loss: 1.8585 | Acc: 71.00% | ETA: 0:01:08 Epoch 033/50 [#############-------] | Loss: 1.7923 | Acc: 73.00% | ETA: 0:01:04 Epoch 034/50 [#############-------] | Loss: 1.7358 | Acc: 78.50% | ETA: 0:01:00 Epoch 035/50 [##############------] | Loss: 1.6550 | Acc: 77.00% | ETA: 0:00:56 Epoch 036/50 [##############------] | Loss: 1.5963 | Acc: 82.00% | ETA: 0:00:52 Epoch 037/50 [##############------] | Loss: 1.4871 | Acc: 87.00% | ETA: 0:00:48 Epoch 038/50 [###############-----] | Loss: 1.4807 | Acc: 83.00% | ETA: 0:00:45 Epoch 039/50 [###############-----] | Loss: 1.3562 | Acc: 85.50% | ETA: 0:00:41 Epoch 040/50 [################----] | Loss: 1.3329 | Acc: 84.50% | ETA: 0:00:37 Epoch 041/50 [################----] | Loss: 1.2510 | Acc: 85.50% | ETA: 0:00:33 Epoch 042/50 [################----] | Loss: 1.2041 | Acc: 88.00% | ETA: 0:00:30 Epoch 043/50 [#################---] | Loss: 1.1408 | Acc: 90.50% | ETA: 0:00:26 Epoch 044/50 [#################---] | Loss: 1.1159 | Acc: 86.50% | ETA: 0:00:22 Epoch 045/50 [##################--] | Loss: 1.0008 | Acc: 91.00% | ETA: 0:00:18 Epoch 046/50 [##################--] | Loss: 0.9149 | Acc: 93.00% | ETA: 0:00:15 Epoch 047/50 [##################--] | Loss: 0.8851 | Acc: 92.00% | ETA: 0:00:11 Epoch 048/50 [###################-] | Loss: 0.7997 | Acc: 94.00% | ETA: 0:00:07 Epoch 049/50 [###################-] | Loss: 0.7989 | Acc: 94.50% | ETA: 0:00:03 Epoch 050/50 [####################] | Loss: 0.7481 | Acc: 94.50% | ETA: 0:00:00 | 0.3 it/s
==================================================
[DONE] Trening završen u: 0:03:07
==================================================

[INFERENCE TEST]
ID uzorka: 135
Stvarno:  id10297
Predviđeno: id10297 (33.90% confidence)
Vrijeme inferencije: 16.75ms
'''
