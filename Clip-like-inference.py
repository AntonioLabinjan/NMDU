import torch
import torchaudio
import torch.nn.functional as F
import os

# --- CONFIG ---
MODEL_PATH = "triplet_run_2/best_triplet_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
EMBEDDING_DIM = 128

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

def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
    
    mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
    db_tf = torchaudio.transforms.AmplitudeToDB()
    spec = db_tf(mel_tf(waveform))
    spec = F.interpolate(spec.unsqueeze(0), size=(128, 128)).to(DEVICE)
    spec = (spec - spec.mean()) / (spec.std() + 1e-7)
    return spec

def compare_voices(file1, file2):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model ne postoji na putanji {MODEL_PATH}")
        return

    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.inference_mode():
        emb1 = model(load_audio(file1))
        emb2 = model(load_audio(file2))
        
        # Euklidska distanca
        distance = torch.pow(emb1 - emb2, 2).sum(1).sqrt().item()
        # Kosinusna sličnost
        similarity = F.cosine_similarity(emb1, emb2).item()

    print(f"\n" + "="*50)
    print(f"🔍 ANALIZA GLASA")
    print(f"="*50)
    # Ispisujemo pune putanje za maksimalnu jasnoću
    print(f"📄 PATH 1: {file1}")
    print(f"📄 PATH 2: {file2}")
    print(f"-"*50)
    print(f"📏 Euclidean distance: {distance:.4f}")
    print(f"🤝 Similarity (0-1):   {similarity:.4f}")
    print(f"-"*50)
    
    # Prag od 0.7-0.8 je obično "sweet spot" za tvoje rezultate
    if distance < 0.75: 
        print("✅ REZULTAT: ISTA OSOBA")
    else:
        print("❌ REZULTAT: RAZLIČITE OSOBE")
    print("="*50 + "\n")

# --- TESTIRANJE ---
# Možeš proslijediti apsolutne ili relativne putanje
f1 = "/home/antonio/Desktop/NMDU/dataset_voice/id10302/OMw6_X2IDvE/00001.wav"
f2 = "/home/antonio/Desktop/NMDU/dataset_voice/id10295/kTLpYHUNf5A/00002.wav"

compare_voices(f1, f2)
