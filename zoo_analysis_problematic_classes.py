import torch
import torchaudio
import torch.nn.functional as F
import os
import glob
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
MODEL_PATH = "triplet_run_2/best_triplet_model.pth"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRODUCTION_THRESHOLD = 0.10

def get_run_folder():
    base = "full_research_report_"
    counter = 1
    while os.path.exists(f"{base}{counter}"): counter += 1
    folder = f"{base}{counter}"
    os.makedirs(folder)
    return folder

class VoiceNetEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        def conv_block(in_f, out_f):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_f), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
            )
        self.features = torch.nn.Sequential(
            conv_block(1, 32), conv_block(32, 64), conv_block(64, 128),
            conv_block(128, 256), conv_block(256, 256)
        )
        self.embedding_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, embedding_dim)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.embedding_head(x)
        return F.normalize(x, p=2, dim=1)

def load_and_preprocess(path):
    try:
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
    except Exception: return None

# --- ANALIZA: ZOO (Modificirana da uključi SVE klase) ---
def run_doddington_zoo_analysis(all_results, RUN_FOLDER):
    print("\n🐐 TEST: Doddington's Zoo Analysis (EER po osobi)...")
    unique_ids = sorted(list(set([r['true'] for r in all_results])))
    per_person_stats = []

    for p_id in unique_ids:
        y_t, y_s = [], []
        for r in all_results:
            if r['true'] == p_id:
                y_t.append(1); y_s.append(-r['dist'])
            elif r['pred_raw'] == p_id:
                y_t.append(0); y_s.append(-r['dist'])

        if len(set(y_t)) > 1:
            fpr, tpr, _ = roc_curve(y_t, y_s)
            eer_p = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            per_person_stats.append({'person_id': p_id, 'eer': eer_p, 'status': 'Evaluated'})
        elif len(set(y_t)) == 1 and list(set(y_t))[0] == 1:
            # Klasa je savršena - nema grešaka
            per_person_stats.append({'person_id': p_id, 'eer': 0.0, 'status': 'Perfect (Sheep)'})
        else:
            # Nema dovoljno podataka za ovu klasu u testu
            per_person_stats.append({'person_id': p_id, 'eer': np.nan, 'status': 'Insufficient Data'})

    df_zoo = pd.DataFrame(per_person_stats).sort_values(by='eer', ascending=False)
    df_zoo.to_csv(f"{RUN_FOLDER}/doddington_zoo_stats.csv", index=False)

    plt.figure(figsize=(14, 7))
    clean_df = df_zoo.dropna(subset=['eer'])
    colors = ['#e74c3c' if x > 0.3 else '#f39c12' if x > 0.15 else '#3498db' for x in clean_df['eer']]
    plt.bar(clean_df['person_id'], clean_df['eer'], color=colors, edgecolor='black')
    plt.axhline(clean_df['eer'].mean(), color='black', ls='--', label='Avg EER')
    plt.xticks(rotation=90); plt.title("Doddington's Zoo (All Classes Included)"); plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/doddington_zoo.png"); plt.close()
    return df_zoo

def run_full_evaluation():
    RUN_FOLDER = get_run_folder()
    print(f"🚀 POKREĆEM TOTALNI IZVJEŠTAJ: {RUN_FOLDER}")

    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    p_ids = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    person_data = {p: sorted([v for v in glob.glob(os.path.join(DATASET_DIR, p, "*")) if os.path.isdir(v)]) for p in p_ids}
    person_data = {k: v for k, v in person_data.items() if len(v) > 1}

    processed_data = {p: [] for p in person_data.keys()}
    all_embs_flat, all_labs_flat = [], []

    with torch.inference_mode():
        for p_id, vids in tqdm(person_data.items(), desc="🧬 Ekstrakcija"):
            for vid_path in vids:
                wavs = glob.glob(os.path.join(vid_path, "*.wav"))
                vid_embs = []
                for w in wavs:
                    spec = load_and_preprocess(w)
                    if spec is not None:
                        emb = model(spec).cpu().numpy().astype('float32')
                        vid_embs.append(emb); all_embs_flat.append(emb); all_labs_flat.append(p_id)
                if vid_embs: processed_data[p_id].append(np.vstack(vid_embs))

    # --- NOVI PRISTUP: Leave-One-Folder-Out za SVAKU osobu ---
    all_results = []
    print("\n🔍 Pokrećem detaljnu unakrsnu validaciju svih klasa...")
    for q_id in tqdm(processed_data.keys(), desc="Analiza klasa"):
        for fold in range(len(processed_data[q_id])):
            g_embs, g_labs = [], []
            q_embs_curr = processed_data[q_id][fold] # Folder koji trenutno testiramo
            
            # Svi ostali folderi (svih ljudi) idu u bazu (gallery)
            for p_id, v_embs in processed_data.items():
                for i, e_set in enumerate(v_embs):
                    if not (p_id == q_id and i == fold):
                        g_embs.append(e_set)
                        g_labs.extend([p_id] * len(e_set))
            
            idx = faiss.IndexFlatL2(EMBEDDING_DIM)
            idx.add(np.vstack(g_embs))
            D, I = idx.search(q_embs_curr, 1)
            
            for i in range(len(q_embs_curr)):
                all_results.append({'true': q_id, 'pred_raw': g_labs[I[i][0]], 'dist': float(D[i][0])})

    # Zoo analiza sada prima rezultate u kojima su sudjelovale SVE klase
    run_doddington_zoo_analysis(all_results, RUN_FOLDER)

    # Standardni t-SNE za vizualni dojam
    X_proj = np.vstack(all_embs_flat)
    tsne_res = TSNE(n_components=2, perplexity=min(30, len(X_proj)-1)).fit_transform(X_proj)
    plt.figure(figsize=(12, 10)); sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=all_labs_flat, palette='tab20', s=30, alpha=0.6, legend=False)
    plt.title("t-SNE Klasteri"); plt.savefig(f"{RUN_FOLDER}/tsne_clusters.png"); plt.close()

    print(f"\n✅ GOTOVO! Provjeri {RUN_FOLDER}/doddington_zoo_stats.csv - sada su svi unutra.")

if __name__ == "__main__":
    run_full_evaluation()
