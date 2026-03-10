import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import random

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_PERSON = 2
EPOCHS = 20

# --- SIMPLE CNN MODEL ---
class VoiceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Nakon dva MaxPool2d(2), 128x128 postaje 32x32
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# --- UTILS ---
def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform

def audio_to_mel(waveform):
    # Mel-spektrogram je standard u prepoznavanju govora (Davis & Mermelstein, 1980)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=256
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    # Interpolacija osigurava fiksnu veličinu ulaza za FC slojeve
    mel_spec = torch.nn.functional.interpolate(
        mel_spec.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False
    )
    return mel_spec.squeeze(0) # Vraća [1, 128, 128]

# --- BUILD DATASET ---
X, y = [], []
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
NUM_CLASSES = len(person_folders)

for label, folder in enumerate(person_folders):
    audio_files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)[:SAMPLES_PER_PERSON]
    for f in audio_files:
        waveform = load_audio(f)
        mel = audio_to_mel(waveform)
        X.append(mel)
        y.append(label)

# ISPRAVAK: stack() već stvara batch dimenziju. 
# Ako mel ima [1, 128, 128], stack(X) daje [Batch, 1, 128, 128]
X = torch.stack(X).to(DEVICE) 
y = torch.tensor(y).to(DEVICE)

# --- TRAINING ---
model = VoiceCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    acc = (outputs.argmax(1) == y).float().mean()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")

# --- TEST ---
model.eval()
with torch.no_grad():
    test_folder = random.choice(person_folders)
    test_file = random.choice(glob.glob(os.path.join(test_folder, "**/*.wav"), recursive=True))
    test_mel = audio_to_mel(load_audio(test_file)).unsqueeze(0).to(DEVICE) # [1, 1, 128, 128]
    output = model(test_mel)
    print(f"Predviđeno: {output.argmax(1).item()} | Stvarno: {person_folders.index(test_folder)}")
