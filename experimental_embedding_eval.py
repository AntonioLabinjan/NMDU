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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# --- CONFIG ---
DATASET_DIR = "dataset_voice"
MODEL_PATH = "triplet_run_2/best_triplet_model.pth"
SAMPLE_RATE = 16000
EMBEDDING_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SIGURNOSNI PARAMETAR ---
PRODUCTION_THRESHOLD = 0.10

def get_run_folder():
    base = "full_eval_report_"
    counter = 1
    while os.path.exists(f"{base}{counter}"): counter += 1
    folder = f"{base}{counter}"
    os.makedirs(folder)
    return folder

# --- MODEL DEFINITION ---
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
        if waveform.shape[0] > 1: 
            waveform = waveform.mean(0, keepdim=True)
        
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512
        )
        db_tf = torchaudio.transforms.AmplitudeToDB()
        spec = db_tf(mel_tf(waveform))
        spec = F.interpolate(spec.unsqueeze(0), size=(128, 128))
        spec = (spec - spec.mean()) / (spec.std() + 1e-7)
        return spec.to(DEVICE)
    except Exception:
        return None

def run_full_evaluation():
    RUN_FOLDER = get_run_folder()
    print(f"🚀 Pokrećem FULL evaluaciju u: {RUN_FOLDER}")

    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Model učitan: {MODEL_PATH}")
    else:
        print(f"❌ ERROR: Model nije pronađen na {MODEL_PATH}")
        return

    model.eval()

    # --- EKSTRAKCIJA PODATAKA ---
    person_ids = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    person_data = {}
    for p_id in person_ids:
        vids = sorted([v for v in glob.glob(os.path.join(DATASET_DIR, p_id, "*")) if os.path.isdir(v)])
        if len(vids) > 1:
            person_data[p_id] = vids

    min_vids = min([len(v) for v in person_data.values()])
    processed_data = {p_id: [] for p_id in person_data.keys()}
    
    all_embs_for_vis = []
    all_labels_for_vis = []

    with torch.inference_mode():
        for p_id, vids in tqdm(person_data.items(), desc="🧬 Ekstrakcija"):
            for vid_path in vids[:min_vids]:
                wavs = glob.glob(os.path.join(vid_path, "*.wav"))
                vid_embs = []
                for w in wavs:
                    spec = load_and_preprocess(w)
                    if spec is not None:
                        emb = model(spec).cpu().numpy().astype('float32')
                        vid_embs.append(emb)
                        all_embs_for_vis.append(emb)
                        all_labels_for_vis.append(p_id)
                
                if vid_embs:
                    processed_data[p_id].append(np.vstack(vid_embs))

    # --- CROSS-VALIDATION ---
    all_results = []
    for fold in range(min_vids):
        gallery_embs, gallery_labels = [], []
        query_embs, query_labels = [], []
        
        for p_id, vids_embs in processed_data.items():
            for i, emb_set in enumerate(vids_embs):
                if i == fold:
                    query_embs.append(emb_set)
                    query_labels.extend([p_id] * len(emb_set))
                else:
                    gallery_embs.append(emb_set)
                    gallery_labels.extend([p_id] * len(emb_set))

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(np.vstack(gallery_embs))
        D, I = index.search(np.vstack(query_embs), 1)
        
        for i in range(len(query_labels)):
            all_results.append({
                'true': query_labels[i], 
                'pred_raw': gallery_labels[I[i][0]], 
                'dist': float(D[i][0])
            })

    # --- METRIKE I SWEEP ---
    thresholds = np.linspace(0.01, 1.0, 100)
    sweep_data = []
    same_dist = [res['dist'] for res in all_results if res['true'] == res['pred_raw']]
    diff_dist = [res['dist'] for res in all_results if res['true'] != res['pred_raw']]

    for t in thresholds:
        y_true, y_pred = [], []
        for res in all_results:
            y_true.append(res['true'])
            y_pred.append(res['pred_raw'] if res['dist'] < t else "unknown")
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        far = sum(1 for d in diff_dist if d < t) / len(diff_dist) if diff_dist else 0
        frr = sum(1 for d in same_dist if d >= t) / len(same_dist) if same_dist else 0
        
        sweep_data.append({
            'Threshold': t, 'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1,
            'FAR': far, 'FRR': frr
        })

    df = pd.DataFrame(sweep_data)
    df.to_csv(f"{RUN_FOLDER}/metrics_sweep.csv", index=False)

    # --- VIZUALIZACIJA 1: PCA & t-SNE (Bez legende za čišći prikaz) ---
    X_vis = np.vstack(all_embs_for_vis)
    num_samples = X_vis.shape[0]
    plt.figure(figsize=(20, 8))
    
    # PCA
    plt.subplot(1, 2, 1)
    pca_res = PCA(n_components=2).fit_transform(X_vis)
    sns.scatterplot(x=pca_res[:,0], y=pca_res[:,1], hue=all_labels_for_vis, palette='tab10', s=40, alpha=0.7, legend=False)
    plt.title("PCA: Projekcija embeddinga (bez legende)")

    # t-SNE
    plt.subplot(1, 2, 2)
    safe_perplexity = min(30, max(1, num_samples - 1))
    try:
        tsne = TSNE(n_components=2, perplexity=safe_perplexity, init='pca', learning_rate='auto')
        tsne_res = tsne.fit_transform(X_vis)
        sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=all_labels_for_vis, palette='tab10', s=40, alpha=0.7, legend=False)
        plt.title(f"t-SNE: Perplexity={safe_perplexity} (bez legende)")
    except Exception as e:
        plt.text(0.5, 0.5, f"t-SNE Error: {e}", ha='center')

    plt.savefig(f"{RUN_FOLDER}/clusters_projection.png")
    plt.close()

    # --- VIZUALIZACIJA 2: DISTANCE & SECURITY ---
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    sns.kdeplot(same_dist, label="Same Person", fill=True, color="green")
    sns.kdeplot(diff_dist, label="Different Person", fill=True, color="red")
    plt.axvline(PRODUCTION_THRESHOLD, color='black', linestyle='--', label=f'Threshold={PRODUCTION_THRESHOLD}')
    plt.title("Distribucija distanci")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['Threshold'], df['FAR'], label='FAR (False Accept)', color='red')
    plt.plot(df['Threshold'], df['FRR'], label='FRR (False Reject)', color='blue')
    plt.axvline(PRODUCTION_THRESHOLD, color='black', linestyle='--')
    plt.title("FAR vs FRR")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(f"{RUN_FOLDER}/security_analysis.png")
    plt.close()

    # --- VIZUALIZACIJA 3: CONFUSION MATRIX ---
    y_true_s, y_pred_s = [], []
    for res in all_results:
        y_true_s.append(res['true'])
        y_pred_s.append(res['pred_raw'] if res['dist'] < PRODUCTION_THRESHOLD else "unknown")

    labels = sorted(list(person_data.keys())) + ["unknown"]
    cm = confusion_matrix(y_true_s, y_pred_s, labels=labels)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels[:-1])
    plt.title(f"Confusion Matrix (T={PRODUCTION_THRESHOLD})")
    plt.savefig(f"{RUN_FOLDER}/confusion_matrix_strict.png")
    plt.close()

    # --- FINAL REPORT ---
    best_row = df.iloc[(df['Threshold'] - PRODUCTION_THRESHOLD).abs().argsort()[:1]]
    print("\n" + "="*50)
    print(f"📊 REPORT ZA PRAG: {PRODUCTION_THRESHOLD}")
    print("="*50)
    print(best_row[['Threshold', 'Accuracy', 'FAR', 'FRR']].to_string(index=False))
    print("-" * 50)
    print(f"✅ Folder: {RUN_FOLDER}")

if __name__ == "__main__":
    run_full_evaluation()
