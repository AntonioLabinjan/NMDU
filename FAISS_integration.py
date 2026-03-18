import torch
import torchaudio
import torch.nn.functional as F
import os
import glob
import faiss
import numpy as np
import pickle
from tqdm import tqdm

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
MODEL_PATH = "triplet_run_2/best_triplet_model.pth" # Prilagodi putanju
INDEX_NAME = "voice_database.index"
METADATA_NAME = "voice_metadata.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
EMBEDDING_DIM = 128

# --- MODEL (Isti kao tvoj) ---
class VoiceNetEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        def conv_block(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_f),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
        self.features = torch.nn.Sequential(
            conv_block(1, 32), conv_block(32, 64), conv_block(64, 128),
            conv_block(128, 256), conv_block(256, 256)
        )
        self.embedding_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_dim)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.embedding_head(x)
        return F.normalize(x, p=2, dim=1)

def load_and_preprocess(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
    
    mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
    db_tf = torchaudio.transforms.AmplitudeToDB()
    spec = db_tf(mel_tf(waveform))
    spec = F.interpolate(spec.unsqueeze(0), size=(128, 128))
    spec = (spec - spec.mean()) / (spec.std() + 1e-7)
    return spec.to(DEVICE)

# --- BUILDING THE INDEX ---
def build_faiss_index():
    # 1. Inicijalizacija modela
    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Baza znanja nije pronađena na {MODEL_PATH}!")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Priprema FAISS indeksa
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = [] # Lista koja mapira index u bazi na info o fileu

    # 3. Iteracija kroz klase (isto kao tvoj folder_map)
    person_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))
    
    print(f"🚀 Krećem u izgradnju indeksa za {len(person_folders)} osoba...")

    all_embeddings = []
    
    with torch.inference_mode():
        for person_path in tqdm(person_folders, desc="Osobe"):
            person_id = os.path.basename(person_path)
            wav_files = glob.glob(os.path.join(person_path, "**/*.wav"), recursive=True)
            
            for f in wav_files:
                spec = load_and_preprocess(f)
                emb = model(spec).cpu().numpy()
                
                all_embeddings.append(emb)
                metadata.append({
                    "person_id": person_id,
                    "file_path": f
                })

    # 4. Dodavanje u FAISS (batch mode)
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings).astype('float32')
        index.add(final_embeddings)
        
        # 5. Spremanje na disk
        faiss.write_index(index, INDEX_NAME)
        with open(METADATA_NAME, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"✅ Indeks uspješno spremljen! Broj zapisa: {index.ntotal}")
    else:
        print("Nema pronađenih podataka.")

# --- SEARCH FUNKCIJA ---
def search_voice(query_file, k=5000):
    # Učitaj resurse
    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    index = faiss.read_index(INDEX_NAME)
    with open(METADATA_NAME, 'rb') as f:
        metadata = pickle.load(f)

    # Procesiraj query
    spec = load_and_preprocess(query_file)
    with torch.inference_mode():
        query_emb = model(spec).cpu().numpy().astype('float32')

    # FAISS Search
    distances, indices = index.search(query_emb, k)

    print(f"\n🔍 Rezultati pretrage za: {os.path.basename(query_file)}")
    print("-" * 50)
    for i in range(k):
        idx = indices[0][i]
        dist = distances[0][i]
        if idx != -1:
            meta = metadata[idx]
            # S obzirom na Triplet Loss i normalizaciju, distanca < 0.6 je obično siguran match
            match_status = "✅ POGODAK" if dist < 0.6 else "❓ SLIČNO"
            print(f"{i+1}. ID: {meta['person_id']} | Dist: {dist:.4f} | {match_status}")
            print(f"   Path: {meta['file_path']}")

if __name__ == "__main__":
    # Prvo izgradi bazu
    build_faiss_index()
    
    # Primjer pretrage
    #test_file = "/home/antonio/Desktop/NMDU/dataset_voice/id10302/OMw6_X2IDvE/00001.wav"
    test_file = "/home/antonio/Desktop/NMDU/dataset_voice/id10302/VRCidrXwd1s/00013.wav"
    search_voice(test_file)
