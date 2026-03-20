"""
VoiceNet Embedding Extractor: Triplet Learning Framework

This script transforms a standard convolutional neural network (CNN) into a 
discriminative feature extractor (embedding model) for voice verification. 

Key architectural and procedural shifts implemented:

1. Architecture Adaptation: The network's classification head (Softmax layer) 
   is replaced with a linear 'Embedding Head' and an L2 normalization layer. 
   Instead of predicting discrete classes, the model maps voice Mel-spectrograms 
   into a 128-dimensional hypersphere where semantic similarity is measured 
   by Euclidean distance.

2. Triplet Loss Paradigm: We shift from Cross-Entropy to Triplet Margin Loss. 
   During training, the model processes triplets: an 'Anchor' (sample A), 
   a 'Positive' (another sample of the same speaker), and a 'Negative' 
   (a sample from a different speaker). 

3. Optimization Objective: The model is trained to minimize the distance 
   between Anchor-Positive pairs while maximizing the distance between 
   Anchor-Negative pairs by at least a predefined 'Margin' (1.0).

4. Feature Transfer: The script supports partial weight loading, allowing 
   the model to inherit spatial feature extraction capabilities from a 
   pretrained classifier ('features' blocks) while fine-tuning the 
   embedding space for verification tasks.

The result is a model that can compare two previously unseen voices and 
determine if they belong to the same person based on their spatial proximity 
in the embedding manifold.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import os
import glob
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
PRETRAINED_PATH = "run_13/best_model.pth" 
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 
EPOCHS = 100 # Povećano jer imamo Early Stopping
LEARNING_RATE = 0.00005 
EMBEDDING_DIM = 128
MARGIN = 1.0

# Early Stopping pragovi
TARGET_ACCURACY = 99.90
TARGET_LOSS = 0.10

def get_run_folder(base_name="triplet_run"):
    n = 1
    while os.path.exists(f"{base_name}_{n}"): n += 1
    run_dir = f"{base_name}_{n}"
    os.makedirs(run_dir)
    return run_dir

RUN_DIR = get_run_folder()

# --- MODEL (VoiceNetEmbedding) ---
class VoiceNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
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
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedding_head(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

# --- TRIPLET DATASET ---
class TripletVoiceDataset(Dataset):
    def __init__(self, folder_map, is_val=False):
        self.folder_map = folder_map
        self.class_ids = list(folder_map.keys())
        self.all_files = []
        for cid in self.class_ids:
            for f in folder_map[cid]:
                self.all_files.append((f, cid))
        
        self.is_val = is_val
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.all_files)

    def _get_spec(self, path, apply_aug=False):
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
        
        if apply_aug:
            shift = int(waveform.shape[1] * 0.1)
            waveform = torch.roll(waveform, random.randint(-shift, shift), dims=1)
            waveform = waveform + 0.005 * torch.randn_like(waveform)

        spec = self.db_transform(self.mel_transform(waveform))
        spec = torch.nn.functional.interpolate(spec.unsqueeze(0), size=(128, 128)).squeeze(0)
        return (spec - spec.mean()) / (spec.std() + 1e-7)

    def __getitem__(self, idx):
        anchor_path, anchor_label = self.all_files[idx]
        
        pos_list = self.folder_map[anchor_label]
        if len(pos_list) > 1:
            pos_path = random.choice([p for p in pos_list if p != anchor_path])
            pos_spec = self._get_spec(pos_path, apply_aug=not self.is_val)
        else:
            pos_spec = self._get_spec(anchor_path, apply_aug=True)
        
        neg_label = random.choice([c for c in self.class_ids if c != anchor_label])
        neg_path = random.choice(self.folder_map[neg_label])
        
        return self._get_spec(anchor_path), pos_spec, self._get_spec(neg_path)

# --- PRIPREMA PODATAKA ---
person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
folder_map = {i: glob.glob(os.path.join(f, "**/*.wav"), recursive=True) for i, f in enumerate(person_folders)}
folder_map = {k: v for k, v in folder_map.items() if len(v) > 0}

train_loader = DataLoader(TripletVoiceDataset(folder_map), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TripletVoiceDataset(folder_map, is_val=True), batch_size=BATCH_SIZE)

# --- INICIJALIZACIJA ---
model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)

if os.path.exists(PRETRAINED_PATH):
    print(f"💉 Učitavam bazu znanja iz {PRETRAINED_PATH}...")
    old_state = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model_dict = model.state_dict()
    pretrained_features = {k: v for k, v in old_state.items() if k.startswith('features')}
    model_dict.update(pretrained_features)
    model.load_state_dict(model_dict)
    print("✅ Konvolucijski slojevi uspješno presuđeni.")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

# --- TRENING PETLJA ---
history = {'epoch': [], 'loss': [], 'val_triplet_acc': []}

print(f"🚀 Krećem s treningom. Cilj: Loss < {TARGET_LOSS} & Acc > {TARGET_ACCURACY}%")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]")
    
    for anchor, pos, neg in pbar:
        anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
        
        optimizer.zero_grad()
        a_emb, p_emb, n_emb = model(anchor), model(pos), model(neg)
        
        loss = criterion(a_emb, p_emb, n_emb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = epoch_loss / len(train_loader)

    # --- EVALUACIJA ---
    model.eval()
    correct_triplets = 0
    total_triplets = 0
    with torch.no_grad():
        for anchor, pos, neg in val_loader:
            anchor, pos, neg = anchor.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            a_emb, p_emb, n_emb = model(anchor), model(pos), model(neg)
            
            dist_pos = (a_emb - p_emb).pow(2).sum(1)
            dist_neg = (a_emb - n_emb).pow(2).sum(1)
            
            correct_triplets += (dist_pos < dist_neg).sum().item()
            total_triplets += anchor.size(0)
    
    t_acc = 100 * correct_triplets / total_triplets
    
    # Logiranje
    history['epoch'].append(epoch + 1)
    history['loss'].append(avg_loss)
    history['val_triplet_acc'].append(t_acc)
    
    print(f"✨ Ep {epoch+1} | Loss: {avg_loss:.4f} | Triplet Accuracy: {t_acc:.2f}%")
    
    # --- EARLY STOPPING PROVJERA ---
    if t_acc >= TARGET_ACCURACY and avg_loss <= TARGET_LOSS:
        print(f"\n🎯 META POGOĐENA! (Acc: {t_acc:.2f}%, Loss: {avg_loss:.4f})")
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_triplet_model.pth"))
        break
    
    # Backup svakih 10 epoha
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(RUN_DIR, f"checkpoint_ep{epoch+1}.pth"))

# --- SPREMANJE I VIZUALIZACIJA ---
pd.DataFrame(history).to_csv(os.path.join(RUN_DIR, "log.csv"), index=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Triplet Loss', color='red')
plt.axhline(y=TARGET_LOSS, color='gray', linestyle='--', label='Target Loss')
plt.title('Gubitak kroz epohe'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['val_triplet_acc'], label='Triplet Accuracy', color='green')
plt.axhline(y=TARGET_ACCURACY, color='gray', linestyle='--', label='Target Acc')
plt.title('Točnost razdvajanja (%)'); plt.legend()

plt.savefig(os.path.join(RUN_DIR, "metrics.png"))
plt.show()

print(f"✅ Trening završen. Sve je u folderu: {RUN_DIR}")
