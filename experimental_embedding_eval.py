import torch
import torchaudio
import torch.nn.functional as F
import os
import glob
import faiss
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATASET_DIR        = "dataset_voice"
MODEL_PATH         = "triplet_run_2/best_triplet_model.pth"
SAMPLE_RATE        = 16000
EMBEDDING_DIM      = 128
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRODUCTION_THRESHOLD = 0.10

# SHOULD: kNN smoothing k
KNN_K = 5

# SHOULD: FAR constraint for operating point
FAR_CONSTRAINT = 0.01


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_run_folder():
    base, counter = "full_eval_report_", 1
    while os.path.exists(f"{base}{counter}"): counter += 1
    folder = f"{base}{counter}"
    os.makedirs(folder)
    return folder


# ─── MODEL ────────────────────────────────────────────────────────────────────
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


# ─── METRIC UTILITIES ─────────────────────────────────────────────────────────
def compute_eer(fpr, tpr, thresholds):
    """Equal Error Rate: point where FAR == FRR."""
    fnr = 1.0 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return eer, eer_threshold


def compute_det_curve(fpr, fnr):
    """Return FPR/FNR in log scale for DET curve."""
    eps = 1e-6
    return np.clip(fpr, eps, 1 - eps), np.clip(fnr, eps, 1 - eps)


def separation_ratio(same_dist, diff_dist):
    """
    Separation Ratio: (mean_diff - mean_same) / (std_same + std_diff)
    Higher → cleaner separation.
    """
    if not same_dist or not diff_dist:
        return float('nan')
    mu_s, std_s = np.mean(same_dist), np.std(same_dist)
    mu_d, std_d = np.mean(diff_dist), np.std(diff_dist)
    denom = std_s + std_d
    return (mu_d - mu_s) / denom if denom > 0 else float('inf')


def knn_smooth_distances(query_embs_flat, gallery_embs_flat, raw_labels, gallery_labels, k=5):
    """
    SHOULD: kNN smoothing.
    For each query, find k nearest gallery neighbours.
    Predicted label = majority vote among k neighbours,
    smoothed distance = mean of top-k distances.
    """
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(gallery_embs_flat)
    D, I = index.search(query_embs_flat, k)

    smoothed = []
    for i in range(len(query_embs_flat)):
        neighbour_labels = [gallery_labels[I[i][j]] for j in range(k)]
        neighbour_dists  = D[i].tolist()
        # majority vote
        from collections import Counter
        pred = Counter(neighbour_labels).most_common(1)[0][0]
        avg_dist = float(np.mean(neighbour_dists))
        smoothed.append((pred, avg_dist))
    return smoothed


def operating_point_at_far(df_sweep, target_far=0.01):
    """
    SHOULD: Find the threshold where FAR ≤ target, pick highest such threshold.
    """
    candidates = df_sweep[df_sweep['FAR'] <= target_far]
    if candidates.empty:
        return None
    row = candidates.iloc[-1]
    return row


def platt_scale(scores, labels):
    """
    NICE: Platt scaling (logistic calibration) on distance scores.
    scores: list of floats (distances), labels: 0=genuine, 1=impostor.
    Returns calibrated probabilities (of being an impostor).
    """
    from sklearn.linear_model import LogisticRegression
    X = np.array(scores).reshape(-1, 1)
    y = np.array(labels)
    lr = LogisticRegression()
    try:
        lr.fit(X, y)
        probs = lr.predict_proba(X)[:, 1]
    except Exception:
        probs = np.zeros(len(scores))
    return probs


def threshold_stability(df_sweep, target_metric='F1', window=5):
    """
    NICE: Measures how 'flat' the top region of a metric curve is.
    Returns (best_threshold, stability_std) — lower std = more stable.
    """
    top_n = df_sweep.nlargest(window, target_metric)
    return top_n['Threshold'].mean(), top_n['Threshold'].std()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run_full_evaluation():
    RUN_FOLDER = get_run_folder()
    print(f"🚀  Full evaluation → {RUN_FOLDER}")

    # Load model
    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅  Model loaded: {MODEL_PATH}")
    else:
        print(f"❌  Model not found at {MODEL_PATH}")
        return
    model.eval()

    # ── DATA EXTRACTION ───────────────────────────────────────────────────────
    person_ids  = sorted([d for d in os.listdir(DATASET_DIR)
                           if os.path.isdir(os.path.join(DATASET_DIR, d))])
    person_data = {}
    for p_id in person_ids:
        vids = sorted([v for v in glob.glob(os.path.join(DATASET_DIR, p_id, "*"))
                       if os.path.isdir(v)])
        if len(vids) > 1:
            person_data[p_id] = vids

    min_vids       = min(len(v) for v in person_data.values())
    processed_data = {p_id: [] for p_id in person_data}
    all_embs_for_vis, all_labels_for_vis = [], []

    # NICE: Latency tracking
    latency_ms_list = []

    with torch.inference_mode():
        for p_id, vids in tqdm(person_data.items(), desc="🧬 Extraction"):
            for vid_path in vids[:min_vids]:
                wavs = glob.glob(os.path.join(vid_path, "*.wav"))
                vid_embs = []
                for w in wavs:
                    spec = load_and_preprocess(w)
                    if spec is not None:
                        t0  = time.perf_counter()
                        emb = model(spec).cpu().numpy().astype('float32')
                        latency_ms_list.append((time.perf_counter() - t0) * 1000)
                        vid_embs.append(emb)
                        all_embs_for_vis.append(emb)
                        all_labels_for_vis.append(p_id)
                if vid_embs:
                    processed_data[p_id].append(np.vstack(vid_embs))

    # ── CROSS-VALIDATION ──────────────────────────────────────────────────────
    all_results = []   # dict per sample: true, pred_raw, dist, knn_pred, knn_dist
    for fold in range(min_vids):
        gallery_embs, gallery_labels = [], []
        query_embs,   query_labels   = [], []

        for p_id, vids_embs in processed_data.items():
            for i, emb_set in enumerate(vids_embs):
                if i == fold:
                    query_embs.append(emb_set);  query_labels.extend([p_id] * len(emb_set))
                else:
                    gallery_embs.append(emb_set); gallery_labels.extend([p_id] * len(emb_set))

        G = np.vstack(gallery_embs)
        Q = np.vstack(query_embs)

        # 1-NN (raw)
        idx1 = faiss.IndexFlatL2(EMBEDDING_DIM)
        idx1.add(G)
        D1, I1 = idx1.search(Q, 1)

        # SHOULD: kNN smoothing
        knn_smoothed = knn_smooth_distances(Q, G, query_labels, gallery_labels, k=min(KNN_K, len(G)))

        for i in range(len(query_labels)):
            all_results.append({
                'true':     query_labels[i],
                'pred_raw': gallery_labels[I1[i][0]],
                'dist':     float(D1[i][0]),
                'knn_pred': knn_smoothed[i][0],
                'knn_dist': knn_smoothed[i][1],
            })

    # ── DISTANCE LISTS ────────────────────────────────────────────────────────
    same_dist = [r['dist'] for r in all_results if r['true'] == r['pred_raw']]
    diff_dist = [r['dist'] for r in all_results if r['true'] != r['pred_raw']]

    # ── THRESHOLD SWEEP ───────────────────────────────────────────────────────
    thresholds  = np.linspace(0.01, 1.0, 200)
    sweep_rows  = []
    for t in thresholds:
        y_true, y_pred = [], []
        for res in all_results:
            y_true.append(res['true'])
            y_pred.append(res['pred_raw'] if res['dist'] < t else "unknown")
        acc   = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        far = sum(1 for d in diff_dist if d < t) / len(diff_dist) if diff_dist else 0
        frr = sum(1 for d in same_dist if d >= t) / len(same_dist) if same_dist else 0
        sweep_rows.append({'Threshold': t, 'Accuracy': acc, 'Precision': p,
                           'Recall': r, 'F1': f1, 'FAR': far, 'FRR': frr})

    df = pd.DataFrame(sweep_rows)
    df.to_csv(f"{RUN_FOLDER}/metrics_sweep.csv", index=False)

    # ── MUST: ROC + AUC ───────────────────────────────────────────────────────
    # Binary: 1 = genuine (same person), 0 = impostor (different person)
    binary_labels = [1 if r['true'] == r['pred_raw'] else 0 for r in all_results]
    # Score for ROC: lower distance → more genuine → negate
    scores_neg    = [-r['dist'] for r in all_results]
    fpr, tpr, roc_thresh = roc_curve(binary_labels, scores_neg)
    roc_auc = auc(fpr, tpr)

    # ── MUST: EER ─────────────────────────────────────────────────────────────
    eer, eer_threshold = compute_eer(fpr, tpr, roc_thresh)

    # ── MUST: DET curve ───────────────────────────────────────────────────────
    fnr          = 1.0 - tpr
    det_fpr, det_fnr = compute_det_curve(fpr, fnr)

    # ── SHOULD: Separation Ratio ──────────────────────────────────────────────
    sep_ratio = separation_ratio(same_dist, diff_dist)

    # ── SHOULD: Operating Point at FAR constraint ─────────────────────────────
    op_row = operating_point_at_far(df, target_far=FAR_CONSTRAINT)

    # ── NICE: Threshold Stability ─────────────────────────────────────────────
    stab_thresh, stab_std = threshold_stability(df, target_metric='F1', window=10)

    # ── NICE: Latency Stats ───────────────────────────────────────────────────
    lat_mean = np.mean(latency_ms_list) if latency_ms_list else 0
    lat_p99  = np.percentile(latency_ms_list, 99) if latency_ms_list else 0

    # ── NICE: Calibration ─────────────────────────────────────────────────────
    impostor_labels = [0 if r['true'] == r['pred_raw'] else 1 for r in all_results]
    raw_dists       = [r['dist'] for r in all_results]
    cal_probs       = platt_scale(raw_dists, impostor_labels)
    frac_pos, mean_pred = calibration_curve(impostor_labels, cal_probs, n_bins=10)

    # ═══════════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ═══════════════════════════════════════════════════════════════════════════

    DARK  = "#0f1117"
    GRID  = "#1e2130"
    ACCENT= "#00e5ff"
    GREEN = "#00ff9f"
    RED   = "#ff4757"
    ORANGE= "#ffa502"
    TEXT  = "#e8eaf6"
    plt.rcParams.update({
        "figure.facecolor": DARK, "axes.facecolor": GRID,
        "axes.edgecolor": "#333", "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": "#2a2d3e",
        "legend.facecolor": GRID, "legend.edgecolor": "#444",
        "font.family": "monospace"
    })

    # ── PLOT 1: Clusters (PCA + t-SNE) ────────────────────────────────────────
    X_vis = np.vstack(all_embs_for_vis)
    num_s = X_vis.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=DARK)
    fig.suptitle("Embedding Space Projections", color=TEXT, fontsize=16, fontweight='bold')

    pca_res = PCA(n_components=2).fit_transform(X_vis)
    unique_labels = sorted(set(all_labels_for_vis))
    palette = sns.color_palette("tab10", n_colors=len(unique_labels))
    label_to_color = {l: palette[i] for i, l in enumerate(unique_labels)}
    colors = [label_to_color[l] for l in all_labels_for_vis]

    axes[0].scatter(pca_res[:,0], pca_res[:,1], c=colors, s=30, alpha=0.7, edgecolors='none')
    axes[0].set_title("PCA Projection", color=TEXT)
    axes[0].grid(True, alpha=0.2)

    safe_perp = min(30, max(1, num_s - 1))
    try:
        tsne_res = TSNE(n_components=2, perplexity=safe_perp, init='pca', learning_rate='auto').fit_transform(X_vis)
        axes[1].scatter(tsne_res[:,0], tsne_res[:,1], c=colors, s=30, alpha=0.7, edgecolors='none')
        axes[1].set_title(f"t-SNE (perplexity={safe_perp})", color=TEXT)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"t-SNE Error:\n{e}", ha='center', va='center', transform=axes[1].transAxes)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/clusters_projection.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 2: MUST — ROC + AUC ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Chance")
    ax.axvline(fpr[np.argmin(np.abs(roc_thresh + eer_threshold))],
               color=GREEN, linestyle=':', alpha=0.6, label=f"EER ≈ {eer:.4f}")
    ax.scatter([fpr[np.argmin(np.abs(fpr - eer))]],
               [tpr[np.argmin(np.abs(fpr - eer))]],
               color=GREEN, s=80, zorder=5)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title(f"ROC Curve  |  AUC = {roc_auc:.4f}  |  EER = {eer:.4f}", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/roc_auc.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 3: MUST — DET Curve ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK)
    ax.plot(det_fpr * 100, det_fnr * 100, color=RED, lw=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title(f"DET Curve  |  EER ≈ {eer*100:.2f}%", fontweight='bold')
    # EER diagonal
    diag = np.linspace(1e-3, 100, 300)
    ax.plot(diag, diag, 'k--', alpha=0.3, label="EER line")
    ax.axvline(eer * 100, color=GREEN, linestyle=':', alpha=0.6, label=f"EER = {eer*100:.2f}%")
    ax.axhline(eer * 100, color=GREEN, linestyle=':', alpha=0.6)
    ax.legend()
    ax.grid(True, alpha=0.2, which='both')
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/det_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 4: MUST — Score Distributions ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=DARK)

    # KDE
    sns.kdeplot(same_dist, ax=axes[0], label="Same Person", fill=True, color=GREEN, alpha=0.5)
    sns.kdeplot(diff_dist, ax=axes[0], label="Different Person", fill=True, color=RED, alpha=0.5)
    axes[0].axvline(PRODUCTION_THRESHOLD, color='white', linestyle='--',
                    label=f"Prod T={PRODUCTION_THRESHOLD}")
    axes[0].axvline(-eer_threshold, color=ORANGE, linestyle=':',
                    label=f"EER T≈{-eer_threshold:.3f}")
    axes[0].set_title(f"Score Distributions  |  Sep Ratio = {sep_ratio:.3f}", fontweight='bold')
    axes[0].set_xlabel("L2 Distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Histogram overlay
    axes[1].hist(same_dist, bins=40, color=GREEN, alpha=0.6, label="Genuine", density=True)
    axes[1].hist(diff_dist, bins=40, color=RED, alpha=0.6, label="Impostor", density=True)
    axes[1].set_title("Score Histograms (Genuine vs Impostor)", fontweight='bold')
    axes[1].set_xlabel("L2 Distance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/score_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 5: FAR/FRR + Operating Points ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK)
    ax.plot(df['Threshold'], df['FAR'], label='FAR (False Accept)', color=RED, lw=2)
    ax.plot(df['Threshold'], df['FRR'], label='FRR (False Reject)', color=ACCENT, lw=2)
    ax.axvline(PRODUCTION_THRESHOLD, color='white', linestyle='--',
               label=f"Prod T={PRODUCTION_THRESHOLD}")
    # EER point
    ax.axvline(-eer_threshold, color=ORANGE, linestyle=':', label=f"EER T≈{-eer_threshold:.3f}")
    # SHOULD: FAR-constrained operating point
    if op_row is not None:
        ax.axvline(op_row['Threshold'], color=GREEN, linestyle='-.',
                   label=f"Op.Point (FAR≤{FAR_CONSTRAINT}) T={op_row['Threshold']:.3f}")
        ax.scatter([op_row['Threshold']], [op_row['FAR']], color=GREEN, s=80, zorder=5)
        ax.scatter([op_row['Threshold']], [op_row['FRR']], color=GREEN, s=80, zorder=5)
    ax.set_title("FAR vs FRR — Operating Points", fontweight='bold')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Error Rate")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/security_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 6: Confusion Matrix ───────────────────────────────────────────────
    y_true_s, y_pred_s = [], []
    for res in all_results:
        y_true_s.append(res['true'])
        y_pred_s.append(res['pred_raw'] if res['dist'] < PRODUCTION_THRESHOLD else "unknown")
    labels = sorted(person_data.keys()) + ["unknown"]
    cm = confusion_matrix(y_true_s, y_pred_s, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=DARK)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels[:-1], ax=ax)
    ax.set_title(f"Confusion Matrix  (T={PRODUCTION_THRESHOLD})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 7: NICE — Calibration Curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK)
    ax.plot(mean_pred, frac_pos, 's-', color=ACCENT, lw=2, label="Platt-calibrated")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability (impostor)")
    ax.set_ylabel("Fraction of Positives (true impostor)")
    ax.set_title("Calibration Curve (Platt Scaling)", fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/calibration_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── PLOT 8: NICE — Latency Distribution ───────────────────────────────────
    if latency_ms_list:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK)
        ax.hist(latency_ms_list, bins=40, color=ACCENT, alpha=0.8, edgecolor='none')
        ax.axvline(lat_mean, color=GREEN, linestyle='--', label=f"Mean = {lat_mean:.2f} ms")
        ax.axvline(lat_p99,  color=RED,   linestyle=':',  label=f"P99  = {lat_p99:.2f} ms")
        ax.set_xlabel("Inference Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Inference Latency Distribution", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{RUN_FOLDER}/latency_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ── PLOT 9: NICE — Threshold Stability (F1 vs Threshold) ─────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    ax.plot(df['Threshold'], df['F1'], color=ACCENT, lw=2, label="F1")
    ax.plot(df['Threshold'], df['Accuracy'], color=GREEN, lw=1.5, linestyle='--', label="Accuracy")
    ax.axvspan(stab_thresh - stab_std, stab_thresh + stab_std,
               alpha=0.15, color=ORANGE, label=f"Stability band (σ={stab_std:.4f})")
    ax.axvline(stab_thresh, color=ORANGE, linestyle=':', label=f"Best T zone ≈ {stab_thresh:.3f}")
    ax.axvline(PRODUCTION_THRESHOLD, color='white', linestyle='--', label=f"Prod T={PRODUCTION_THRESHOLD}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Stability  (F1 peak σ={stab_std:.4f} → {'STABLE' if stab_std < 0.05 else 'UNSTABLE'})",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/threshold_stability.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════════════════════════════════════════
    #  FINAL REPORT (console + CSV)
    # ═══════════════════════════════════════════════════════════════════════════
    best_row = df.iloc[(df['Threshold'] - PRODUCTION_THRESHOLD).abs().argsort().iloc[0]]

    report_lines = [
        "=" * 60,
        f"  FULL EVALUATION REPORT  |  {RUN_FOLDER}",
        "=" * 60,
        "",
        "── MUST: Core Biometric Metrics ──",
        f"  ROC AUC             : {roc_auc:.4f}",
        f"  EER                 : {eer*100:.2f}%  (threshold ≈ {-eer_threshold:.4f})",
        "",
        f"── At Production Threshold ({PRODUCTION_THRESHOLD}) ──",
        f"  Accuracy            : {float(best_row['Accuracy']):.4f}",
        f"  FAR                 : {float(best_row['FAR']):.4f}",
        f"  FRR                 : {float(best_row['FRR']):.4f}",
        f"  F1                  : {float(best_row['F1']):.4f}",
        "",
        "── SHOULD: Additional Analysis ──",
        f"  Separation Ratio    : {sep_ratio:.3f}  (higher = better)",
        f"  kNN Smoothing k     : {KNN_K}",
    ]
    if op_row is not None:
        report_lines += [
            f"  Op.Point (FAR≤{FAR_CONSTRAINT}) : T={op_row['Threshold']:.4f}, "
            f"FAR={op_row['FAR']:.4f}, FRR={op_row['FRR']:.4f}"
        ]
    else:
        report_lines.append(f"  Op.Point (FAR≤{FAR_CONSTRAINT}) : not achievable in sweep range")

    report_lines += [
        "",
        "── NICE: Operational Metrics ──",
        f"  Inference Latency   : mean={lat_mean:.2f} ms, p99={lat_p99:.2f} ms",
        f"  Threshold Stability : best_zone≈{stab_thresh:.4f}, σ={stab_std:.4f} "
            f"({'STABLE' if stab_std < 0.05 else 'UNSTABLE'})",
        "  Calibration         : Platt scaling applied → calibration_curve.png",
        "",
        "── Output Files ──",
        "  clusters_projection.png",
        "  roc_auc.png",
        "  det_curve.png",
        "  score_distributions.png",
        "  security_analysis.png",
        "  confusion_matrix.png",
        "  calibration_curve.png",
        "  latency_distribution.png",
        "  threshold_stability.png",
        "  metrics_sweep.csv",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open(f"{RUN_FOLDER}/report.txt", "w") as f:
        f.write(report_text)

    # Summary CSV
    summary = {
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": -eer_threshold,
        "separation_ratio": sep_ratio,
        "prod_threshold": PRODUCTION_THRESHOLD,
        "prod_accuracy": float(best_row['Accuracy']),
        "prod_far":      float(best_row['FAR']),
        "prod_frr":      float(best_row['FRR']),
        "prod_f1":       float(best_row['F1']),
        "lat_mean_ms": lat_mean,
        "lat_p99_ms": lat_p99,
        "stability_threshold": stab_thresh,
        "stability_std": stab_std,
    }
    if op_row is not None:
        summary["op_threshold"]  = float(op_row['Threshold'])
        summary["op_far"]        = float(op_row['FAR'])
        summary["op_frr"]        = float(op_row['FRR'])

    pd.DataFrame([summary]).to_csv(f"{RUN_FOLDER}/summary.csv", index=False)
    print(f"\n✅  Evaluation complete → {RUN_FOLDER}/")


if __name__ == "__main__":
    run_full_evaluation()
