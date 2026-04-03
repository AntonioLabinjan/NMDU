import torch
import torchaudio
import torch.nn.functional as F
import os
import glob
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
MODEL_PATH = "triplet_run_2/best_triplet_model.pth"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_run_folder():
    base = "threshold_sweep_"
    counter = 1
    while os.path.exists(f"{base}{counter}"): counter += 1
    folder = f"{base}{counter}"
    os.makedirs(folder)
    return folder

RUN_FOLDER = get_run_folder()

# --- MODEL ---
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
        if sr != SAMPLE_RATE: waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
        mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512)
        db_tf = torchaudio.transforms.AmplitudeToDB()
        spec = db_tf(mel_tf(waveform))
        spec = F.interpolate(spec.unsqueeze(0), size=(128, 128))
        spec = (spec - spec.mean()) / (spec.std() + 1e-7)
        return spec.to(DEVICE)
    except: return None

def sweep_thresholds():
    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"🚀 Pokrećem Full Sweep u: {RUN_FOLDER}")

    person_ids = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    person_data = {p_id: v for p_id in person_ids if len(v := sorted([v for v in glob.glob(os.path.join(DATASET_DIR, p_id, "*")) if os.path.isdir(v)])) > 1}
    min_vids = min([len(v) for v in person_data.values()])
    processed_data = {p_id: [] for p_id in person_data.keys()}
    
    with torch.inference_mode():
        for p_id, vids in tqdm(person_data.items(), desc="🧬 Ekstrakcija"):
            for vid_path in vids[:min_vids]:
                wavs = glob.glob(os.path.join(vid_path, "*.wav"))
                embs = [model(load_and_preprocess(w)).cpu().numpy().astype('float32') for w in wavs if load_and_preprocess(w) is not None]
                if embs: processed_data[p_id].append(np.vstack(embs))

    all_results = []
    for fold in range(min_vids):
        gallery_embs, gallery_labels = [], []
        query_embs, query_labels = [], []
        for p_id, vids_embs in processed_data.items():
            for i, emb_set in enumerate(vids_embs):
                if i == fold: query_embs.append(emb_set); query_labels.extend([p_id] * len(emb_set))
                else: gallery_embs.append(emb_set); gallery_labels.extend([p_id] * len(emb_set))

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(np.vstack(gallery_embs))
        D, I = index.search(np.vstack(query_embs), 1)
        for i in range(len(query_labels)):
            all_results.append({'true': query_labels[i], 'pred_raw': gallery_labels[I[i][0]], 'dist': D[i][0]})

    thresholds = np.arange(0.1, 1.1, 0.1)
    sweep_data = []
    best_f1 = -1
    best_t = 0.3

    for t in thresholds:
        y_true, y_pred = [], []
        for res in all_results:
            y_true.append(res['true'])
            y_pred.append(res['pred_raw'] if res['dist'] < t else "unknown")
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
        
        sweep_data.append({
            'Threshold': round(t, 2), 'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1,
            'TP': sum(1 for gt, pr in zip(y_true, y_pred) if gt == pr and pr != "unknown"),
            'FP': sum(1 for gt, pr in zip(y_true, y_pred) if gt != pr and pr != "unknown"),
            'FN': sum(1 for gt, pr in zip(y_true, y_pred) if pr == "unknown")
        })

    # --- SAVE CSV & PLOTS ---
    df = pd.DataFrame(sweep_data)
    print("\n" + df.to_string(index=False))
    df.to_csv(f"{RUN_FOLDER}/sweep_results.csv", index=False)

    # Plot performance
    plt.figure(figsize=(10, 5))
    plt.plot(df['Threshold'], df['Accuracy'], label='Accuracy', marker='o')
    plt.plot(df['Threshold'], df['F1'], label='F1 Score', marker='s')
    plt.title(f"Performance Sweep (Best T: {round(best_t, 2)})")
    plt.grid(True); plt.legend(); plt.savefig(f"{RUN_FOLDER}/performance_sweep.png")

    # --- PLOT CONFUSION MATRIX FOR BEST THRESHOLD ---
    y_true_best, y_pred_best = [], []
    for res in all_results:
        y_true_best.append(res['true'])
        y_pred_best.append(res['pred_raw'] if res['dist'] < best_t else "unknown")

    # Uzmi sve ID-ove da graf ne bude pretrpan
    top_ids = sorted(list(person_data.keys()))
    cm = confusion_matrix(y_true_best, y_pred_best, labels=top_ids + ["unknown"])
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_ids + ["unknown"], yticklabels=top_ids)
    plt.title(f"Confusion Matrix for Best Threshold ({round(best_t, 2)})")
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig(f"{RUN_FOLDER}/best_threshold_cm.png")
    
    print(f"\n✅ Gotovo! Najbolji prag {round(best_t, 2)} spremljen u matricu.")

if __name__ == "__main__":
    sweep_thresholds()
