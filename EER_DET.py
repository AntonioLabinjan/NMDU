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

# --- SIGURNOSNI PARAMETAR ---
PRODUCTION_THRESHOLD = 0.10

def get_run_folder():
    base = "full_research_report_"
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

# --- NEW: BIOMETRIC DIAGNOSTICS (EER & DET) ---
def run_biometric_diagnostics(all_results, RUN_FOLDER):
    print("\n📊 TEST 3/3: Biometrijska dijagnostika (EER/DET)...")
    
    # Ground truth: 1 ako je ista osoba, 0 ako je uljez/druga osoba
    y_true = [1 if r['true'] == r['pred_raw'] else 0 for r in all_results]
    # Scoreovi (negativna distanca jer roc_curve očekuje da veći score znači veću sličnost)
    y_scores = [-r['dist'] for r in all_results] 

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Izračun EER-a (točka gdje je False Acceptance Rate == False Rejection Rate)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh_eer = -interp1d(fpr, thresholds)(eer) # Vraćamo u pozitivnu L2 distancu

    # Plot 1: DET Krivulja (Log-Log skala)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, fnr, label=f'VoiceNet (EER = {eer:.4f})', lw=2, color='darkblue')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('False Rejection Rate (FRR)')
    plt.title('DET Krivulja (Detection Error Tradeoff)')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(f"{RUN_FOLDER}/det_curve.png")
    plt.close()

    # Plot 2: FAR vs FRR po pragu
    plt.figure(figsize=(10, 5))
    dist_thresholds = [-t for t in thresholds]
    plt.plot(dist_thresholds, fpr, label='FAR (Lažni upadi)', color='red')
    plt.plot(dist_thresholds, fnr, label='FRR (Lažna odbijanja)', color='blue')
    plt.axvline(thresh_eer, color='black', ls='--', label=f'EER Prag ({thresh_eer:.3f})')
    plt.xlabel('L2 Distance Threshold')
    plt.ylabel('Error Rate')
    plt.title('Balans sigurnosti (FAR vs FRR)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{RUN_FOLDER}/far_frr_tradeoff.png")
    plt.close()

    return eer, thresh_eer

# --- TEST 1: TIME SCALABILITY ---
def run_time_scalability_test(processed_data, model, RUN_FOLDER):
    print("\n⏱️ TEST 1/3: Vremenska skalabilnost (Latency)...")
    all_p_ids = sorted(list(processed_data.keys()))
    time_results = []
    
    dummy_input = torch.randn(1, 1, 128, 128).to(DEVICE)
    with torch.inference_mode():
        for _ in range(20): _ = model(dummy_input)
        start_inf = time.perf_counter()
        for _ in range(100): _ = model(dummy_input)
        avg_inf = (time.perf_counter() - start_inf) / 100

    for n in range(1, len(all_p_ids) + 1):
        curr_p_ids = all_p_ids[:n]
        gallery_list = [np.vstack(processed_data[p][1:]) for p in curr_p_ids]
        X_gallery = np.vstack(gallery_list)
        X_query = processed_data[all_p_ids[0]][0][0:1] 

        s_idx = time.perf_counter()
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(X_gallery)
        idx_t = time.perf_counter() - s_idx

        s_src = time.perf_counter()
        for _ in range(100): index.search(X_query, 1)
        src_t = (time.perf_counter() - s_src) / 100

        time_results.append({
            'num_classes': n, 'num_vectors': len(X_gallery),
            'inference_ms': avg_inf * 1000, 'indexing_ms': idx_t * 1000,
            'search_ms': src_t * 1000, 'total_latency_ms': (avg_inf + src_t) * 1000
        })

    df_time = pd.DataFrame(time_results)
    df_time.to_csv(f"{RUN_FOLDER}/time_scalability.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(df_time['num_classes'], df_time['search_ms'], 'r-o', label='Search Latency (ms)')
    plt.plot(df_time['num_classes'], df_time['total_latency_ms'], 'g--', label='Total Latency (Inf+Search)')
    plt.xlabel('Broj klasa'); plt.ylabel('Vrijeme (ms)'); plt.title('Vremenska složenost sustava'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{RUN_FOLDER}/time_complexity.png"); plt.close()

# --- TEST 2: ACCURACY SCALABILITY ---
def run_accuracy_scalability_test(processed_data, RUN_FOLDER):
    print("\n📈 TEST 2/3: Skalabilnost točnosti...")
    all_p_ids = sorted(list(processed_data.keys()))
    scalability_results = []
    
    for n in range(1, len(all_p_ids) + 1):
        curr_data = {p: processed_data[p] for p in all_p_ids[:n]}
        fold_results = []
        min_vids = len(next(iter(curr_data.values())))
        
        for fold in range(min_vids):
            g_embs, g_labs, q_embs, q_labs = [], [], [], []
            for p_id, v_embs in curr_data.items():
                for i, e_set in enumerate(v_embs):
                    if i == fold: q_embs.append(e_set); q_labs.extend([p_id]*len(e_set))
                    else: g_embs.append(e_set); g_labs.extend([p_id]*len(e_set))
            
            index = faiss.IndexFlatL2(EMBEDDING_DIM); index.add(np.vstack(g_embs))
            D, I = index.search(np.vstack(q_embs), 1)
            for i in range(len(q_labs)):
                dist = float(D[i][0])
                pred = g_labs[I[i][0]] if dist < PRODUCTION_THRESHOLD else "unknown"
                fold_results.append({'true': q_labs[i], 'pred': pred})

        y_t, y_p = [r['true'] for r in fold_results], [r['pred'] for r in fold_results]
        acc = accuracy_score(y_t, y_p)
        _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average='weighted', zero_division=0)
        scalability_results.append({'num_classes': n, 'accuracy': acc, 'f1': f1})
        if n % 5 == 0 or n == 1: print(f" > Progres: {n} klasa | Acc: {acc:.4f}")

    df_acc = pd.DataFrame(scalability_results)
    df_acc.to_csv(f"{RUN_FOLDER}/accuracy_scalability.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(df_acc['num_classes'], df_acc['accuracy'], 'b-s', label='Accuracy')
    plt.plot(df_acc['num_classes'], df_acc['f1'], 'm-o', label='F1 Score')
    plt.xlabel('Broj osoba'); plt.ylabel('Score'); plt.title('Pad performansi s povećanjem baze'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{RUN_FOLDER}/accuracy_scalability.png"); plt.close()

# --- MAIN EXECUTION ---
def run_full_evaluation():
    RUN_FOLDER = get_run_folder()
    print(f"🚀 POKREĆEM TOTALNI IZVJEŠTAJ: {RUN_FOLDER}")

    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model ne postoji na {MODEL_PATH}"); return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    p_ids = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    person_data = {p: sorted([v for v in glob.glob(os.path.join(DATASET_DIR, p, "*")) if os.path.isdir(v)]) for p in p_ids}
    person_data = {k: v for k, v in person_data.items() if len(v) > 1}

    processed_data = {p: [] for p in person_data.keys()}
    all_embs_flat, all_labs_flat = [], []

    with torch.inference_mode():
        for p_id, vids in tqdm(person_data.items(), desc="🧬 Ekstrakcija embeddinga"):
            for vid_path in vids:
                wavs = glob.glob(os.path.join(vid_path, "*.wav"))
                vid_embs = []
                for w in wavs:
                    spec = load_and_preprocess(w)
                    if spec is not None:
                        emb = model(spec).cpu().numpy().astype('float32')
                        vid_embs.append(emb)
                        all_embs_flat.append(emb)
                        all_labs_flat.append(p_id)
                if vid_embs:
                    processed_data[p_id].append(np.vstack(vid_embs))

    # 1. Scalability & Latency Tests
    run_time_scalability_test(processed_data, model, RUN_FOLDER)
    run_accuracy_scalability_test(processed_data, RUN_FOLDER)

    # 2. Collect Results for Biometrics
    all_results = []
    min_vids = min([len(v) for v in processed_data.values()])
    for fold in range(min_vids):
        g_embs, g_labs, q_embs, q_labs = [], [], [], []
        for p_id, v_embs in processed_data.items():
            for i, e_set in enumerate(v_embs):
                if i == fold: q_embs.append(e_set); q_labs.extend([p_id]*len(e_set))
                else: g_embs.append(e_set); g_labs.extend([p_id]*len(e_set))
        idx = faiss.IndexFlatL2(EMBEDDING_DIM); idx.add(np.vstack(g_embs))
        D, I = idx.search(np.vstack(q_embs), 1)
        for i in range(len(q_labs)):
            all_results.append({'true': q_labs[i], 'pred_raw': g_labs[I[i][0]], 'dist': float(D[i][0])})

    # 3. Biometric Analysis (EER/DET)
    eer, opt_thresh = run_biometric_diagnostics(all_results, RUN_FOLDER)

    # 4. Standard Visualizations
    same_d = [r['dist'] for r in all_results if r['true'] == r['pred_raw']]
    diff_d = [r['dist'] for r in all_results if r['true'] != r['pred_raw']]
    plt.figure(figsize=(10, 5))
    sns.kdeplot(same_d, label="Same Person", fill=True, color="green")
    sns.kdeplot(diff_d, label="Different Person", fill=True, color="red")
    plt.axvline(opt_thresh, color='black', ls='--', label=f'EER Threshold ({opt_thresh:.3f})')
    plt.title("Distribucija distanci s EER pragom"); plt.legend(); plt.savefig(f"{RUN_FOLDER}/distance_dist.png"); plt.close()

    X_proj = np.vstack(all_embs_flat)
    tsne_res = TSNE(n_components=2, perplexity=min(30, len(X_proj)-1)).fit_transform(X_proj)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=all_labs_flat, palette='tab20', s=30, alpha=0.6, legend=False)
    plt.title("t-SNE Klasteri"); plt.savefig(f"{RUN_FOLDER}/tsne_clusters.png"); plt.close()

    y_true_f = [r['true'] for r in all_results]
    y_pred_f = [r['pred_raw'] if r['dist'] < PRODUCTION_THRESHOLD else "unknown" for r in all_results]
    labels = sorted(list(processed_data.keys())) + ["unknown"]
    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels[:-1])
    plt.title("Confusion Matrix Heatmap"); plt.savefig(f"{RUN_FOLDER}/confusion_matrix.png"); plt.close()

    print(f"\n✅ SVE GOTOVO! Izvještaj je spreman u: {RUN_FOLDER}")
    print(f"📈 Equal Error Rate (EER): {eer:.4%}")
    print(f"🎯 Optimalni Prag za EER: {opt_thresh:.4f}")

if __name__ == "__main__":
    run_full_evaluation()
