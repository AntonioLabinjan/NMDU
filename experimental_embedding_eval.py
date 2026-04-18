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
from collections import Counter
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
DATASET_DIR          = "dataset_voice"
MODEL_PATH           = "triplet_run_2/best_triplet_model.pth"
SAMPLE_RATE          = 16000
EMBEDDING_DIM        = 128
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Legacy single-threshold (kept for comparison plots) ──────────────────────
PRODUCTION_THRESHOLD = 0.10

# ── Open-set K1/K2 consensus parameters ──────────────────────────────────────
# K2: neighbourhood retrieval size  (how many FAISS hits to fetch)
# K1: minimum votes required to ACCEPT  (must be < K2)
# DIST_GATE: maximum allowed mean distance for winning class (hard reject above)
K2          = 9       # retrieve top-9 neighbors
K1          = 6       # require 6/9 votes for accept  (~67% consensus)
DIST_GATE   = 0.35    # distance ceiling even if votes pass

# ── FAR operating-point constraint ───────────────────────────────────────────
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
        spec  = db_tf(mel_tf(waveform))
        spec  = F.interpolate(spec.unsqueeze(0), size=(128, 128))
        spec  = (spec - spec.mean()) / (spec.std() + 1e-7)
        return spec.to(DEVICE)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  OPEN-SET K1/K2 CONSENSUS DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def consensus_decision(distances: np.ndarray,
                        indices: np.ndarray,
                        gallery_labels: list,
                        k1: int,
                        dist_gate: float,
                        weighted: bool = True):
    """
    Open-set decision for a single query using K1/K2 consensus.

    Parameters
    ----------
    distances    : (K2,) L2 distances to K2 nearest gallery neighbors
    indices      : (K2,) gallery indices
    gallery_labels : flat list of all gallery labels
    k1           : minimum vote count to accept
    dist_gate    : maximum mean distance for the winning class (hard reject)
    weighted     : use inverse-distance weighting instead of raw vote counts

    Returns
    -------
    pred         : predicted identity str, or "unknown"
    confidence   : float in [0, 1] — vote share of winning class (entropy-based)
    winning_dist : mean L2 distance of winning-class neighbors
    vote_counts  : dict {label: weighted_votes}
    """
    k2 = len(indices)
    labels_k2 = [gallery_labels[idx] for idx in indices]

    if weighted:
        # Inverse-distance weights; clip to avoid division by zero
        eps     = 1e-6
        weights = 1.0 / (distances + eps)
        weights /= weights.sum()                      # normalize to sum=1
        vote_dict: dict[str, float] = {}
        for lbl, w in zip(labels_k2, weights):
            vote_dict[lbl] = vote_dict.get(lbl, 0.0) + w
    else:
        vote_dict = {lbl: labels_k2.count(lbl) for lbl in set(labels_k2)}

    # Winning class
    winning_label = max(vote_dict, key=vote_dict.get)
    winning_votes = vote_dict[winning_label]
    total_votes   = sum(vote_dict.values())

    # Raw vote count for K1 check (always integer count, not weighted)
    raw_count = labels_k2.count(winning_label)

    # Mean distance of winning-class neighbors
    winning_dists = [distances[j] for j, lbl in enumerate(labels_k2) if lbl == winning_label]
    winning_dist  = float(np.mean(winning_dists))

    # Confidence: vote share (weighted)
    confidence = float(winning_votes / total_votes) if total_votes > 0 else 0.0

    # ── Two-condition ACCEPT ──────────────────────────────────────────────────
    # Condition 1: raw vote majority ≥ K1
    # Condition 2: mean distance of winning class ≤ dist_gate
    if raw_count >= k1 and winning_dist <= dist_gate:
        return winning_label, confidence, winning_dist, vote_dict
    else:
        return "unknown", confidence, winning_dist, vote_dict


def run_consensus_on_fold(Q: np.ndarray,
                           G: np.ndarray,
                           query_labels: list,
                           gallery_labels: list,
                           k2: int,
                           k1: int,
                           dist_gate: float,
                           weighted: bool = True) -> list:
    """
    Runs K1/K2 consensus for all queries in one fold.
    Returns list of result dicts.
    """
    actual_k2 = min(k2, len(G))
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(G)
    D, I = index.search(Q, actual_k2)     # (N, k2) each

    results = []
    for i in range(len(query_labels)):
        pred, conf, w_dist, votes = consensus_decision(
            distances      = D[i],
            indices        = I[i],
            gallery_labels = gallery_labels,
            k1             = k1,
            dist_gate      = dist_gate,
            weighted       = weighted,
        )
        results.append({
            'true':       query_labels[i],
            'pred':       pred,
            'confidence': conf,
            'win_dist':   w_dist,
            'top1_dist':  float(D[i][0]),           # kept for legacy comparison
            'top1_label': gallery_labels[I[i][0]],  # kept for legacy comparison
            'votes':      votes,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eer(fpr, tpr, thresholds):
    fnr      = 1.0 - tpr
    idx      = np.argmin(np.abs(fpr - fnr))
    eer      = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]


def compute_det_curve(fpr, fnr):
    eps = 1e-6
    return np.clip(fpr, eps, 1 - eps), np.clip(fnr, eps, 1 - eps)


def separation_ratio(same_dist, diff_dist):
    if not same_dist or not diff_dist:
        return float('nan')
    mu_s, std_s = np.mean(same_dist), np.std(same_dist)
    mu_d, std_d = np.mean(diff_dist), np.std(diff_dist)
    denom = std_s + std_d
    return (mu_d - mu_s) / denom if denom > 0 else float('inf')


def operating_point_at_far(df_sweep, target_far=0.01):
    candidates = df_sweep[df_sweep['FAR'] <= target_far]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def platt_scale(scores, labels):
    from sklearn.linear_model import LogisticRegression
    X  = np.array(scores).reshape(-1, 1)
    y  = np.array(labels)
    lr = LogisticRegression()
    try:
        lr.fit(X, y)
        return lr.predict_proba(X)[:, 1]
    except Exception:
        return np.zeros(len(scores))


def threshold_stability(df_sweep, target_metric='F1', window=10):
    top_n = df_sweep.nlargest(window, target_metric)
    return top_n['Threshold'].mean(), top_n['Threshold'].std()


def sweep_k1_k2(processed_data, person_data, min_vids, k2_range, k1_fracs):
    """
    Grid search over K2 sizes and K1 fractions.
    Returns a DataFrame with FAR / FRR / F1 per (K2, K1, dist_gate) combination.
    Uses the same dist_gate as the global DIST_GATE for simplicity.
    """
    rows = []
    for k2_val in k2_range:
        for k1_frac in k1_fracs:
            k1_val = max(1, int(np.ceil(k1_frac * k2_val)))
            fold_results = []
            for fold in range(min_vids):
                gallery_embs, gallery_labels = [], []
                query_embs,   query_labels   = [], []
                for p_id, vids_embs in processed_data.items():
                    for i, emb_set in enumerate(vids_embs):
                        if i == fold:
                            query_embs.append(emb_set)
                            query_labels.extend([p_id] * len(emb_set))
                        else:
                            gallery_embs.append(emb_set)
                            gallery_labels.extend([p_id] * len(emb_set))
                G = np.vstack(gallery_embs)
                Q = np.vstack(query_embs)
                fold_results += run_consensus_on_fold(
                    Q, G, query_labels, gallery_labels,
                    k2=k2_val, k1=k1_val, dist_gate=DIST_GATE, weighted=True
                )
            # compute metrics
            y_true = [r['true'] for r in fold_results]
            y_pred = [r['pred'] for r in fold_results]
            known_mask = [r['pred'] != 'unknown' for r in fold_results]
            genuine_dists = [r['win_dist'] for r in fold_results if r['true'] == r['top1_label']]
            impostor_dists= [r['win_dist'] for r in fold_results if r['true'] != r['top1_label']]
            far = (sum(1 for r in fold_results if r['true'] != r['pred'] and r['pred'] != 'unknown')
                   / max(1, len([r for r in fold_results if r['true'] != r['top1_label']])))
            frr = (sum(1 for r in fold_results if r['pred'] == 'unknown' and r['true'] == r['top1_label'])
                   / max(1, len([r for r in fold_results if r['true'] == r['top1_label']])))
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            rows.append({'K2': k2_val, 'K1': k1_val, 'K1_frac': k1_frac,
                         'FAR': far, 'FRR': frr, 'F1': f1})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation():
    RUN_FOLDER = get_run_folder()
    print(f"🚀  Full evaluation (open-set K1/K2) → {RUN_FOLDER}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅  Model loaded: {MODEL_PATH}")
    else:
        print(f"❌  Model not found: {MODEL_PATH}"); return
    model.eval()

    # ── Data extraction ───────────────────────────────────────────────────────
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
    latency_ms_list = []

    with torch.inference_mode():
        for p_id, vids in tqdm(person_data.items(), desc="🧬 Extraction"):
            for vid_path in vids[:min_vids]:
                wavs     = glob.glob(os.path.join(vid_path, "*.wav"))
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

    # ── Cross-validation with K1/K2 consensus ────────────────────────────────
    all_results = []
    for fold in range(min_vids):
        gallery_embs, gallery_labels = [], []
        query_embs,   query_labels   = [], []

        for p_id, vids_embs in processed_data.items():
            for i, emb_set in enumerate(vids_embs):
                if i == fold:
                    query_embs.append(emb_set)
                    query_labels.extend([p_id] * len(emb_set))
                else:
                    gallery_embs.append(emb_set)
                    gallery_labels.extend([p_id] * len(emb_set))

        G = np.vstack(gallery_embs)
        Q = np.vstack(query_embs)

        fold_res = run_consensus_on_fold(
            Q, G, query_labels, gallery_labels,
            k2=K2, k1=K1, dist_gate=DIST_GATE, weighted=True
        )
        all_results.extend(fold_res)

    # ── Distance lists for analysis ───────────────────────────────────────────
    # "genuine" = top-1 label matches true label (regardless of consensus decision)
    same_dist = [r['top1_dist'] for r in all_results if r['true'] == r['top1_label']]
    diff_dist = [r['top1_dist'] for r in all_results if r['true'] != r['top1_label']]

    # ── Legacy threshold sweep (for comparison plots) ─────────────────────────
    thresholds = np.linspace(0.01, 1.0, 200)
    sweep_rows = []
    for t in thresholds:
        y_true, y_pred_t = [], []
        for res in all_results:
            y_true.append(res['true'])
            y_pred_t.append(res['top1_label'] if res['top1_dist'] < t else "unknown")
        acc          = accuracy_score(y_true, y_pred_t)
        p, r, f1, _  = precision_recall_fscore_support(y_true, y_pred_t, average='weighted', zero_division=0)
        far          = sum(1 for d in diff_dist if d < t) / max(1, len(diff_dist))
        frr          = sum(1 for d in same_dist if d >= t) / max(1, len(same_dist))
        sweep_rows.append({'Threshold': t, 'Accuracy': acc, 'Precision': p,
                           'Recall': r, 'F1': f1, 'FAR': far, 'FRR': frr})
    df_legacy = pd.DataFrame(sweep_rows)
    df_legacy.to_csv(f"{RUN_FOLDER}/legacy_threshold_sweep.csv", index=False)

    # ── Consensus system performance ──────────────────────────────────────────
    y_true_c  = [r['true']  for r in all_results]
    y_pred_c  = [r['pred']  for r in all_results]

    # FAR: impostor accepted as someone (pred != unknown AND pred != true)
    impostors      = [r for r in all_results if r['true'] != r['top1_label']]
    consensus_far  = sum(1 for r in impostors if r['pred'] != 'unknown') / max(1, len(impostors))
    # FRR: genuine rejected (pred == unknown AND top1 was correct class)
    genuines       = [r for r in all_results if r['true'] == r['top1_label']]
    consensus_frr  = sum(1 for r in genuines  if r['pred'] == 'unknown')  / max(1, len(genuines))
    consensus_acc  = accuracy_score(y_true_c, y_pred_c)
    _, _, consensus_f1, _ = precision_recall_fscore_support(
        y_true_c, y_pred_c, average='weighted', zero_division=0
    )
    unknown_rate = sum(1 for r in all_results if r['pred'] == 'unknown') / len(all_results)

    # ── ROC / AUC / EER (on top-1 distances, binary genuine/impostor) ─────────
    binary_labels = [1 if r['true'] == r['top1_label'] else 0 for r in all_results]
    scores_neg    = [-r['top1_dist'] for r in all_results]
    fpr, tpr, roc_thresh = roc_curve(binary_labels, scores_neg)
    roc_auc              = auc(fpr, tpr)
    eer, eer_threshold   = compute_eer(fpr, tpr, roc_thresh)

    # DET
    fnr = 1.0 - tpr
    det_fpr, det_fnr = compute_det_curve(fpr, fnr)

    # Separation ratio
    sep_ratio = separation_ratio(same_dist, diff_dist)

    # Operating point
    op_row = operating_point_at_far(df_legacy, target_far=FAR_CONSTRAINT)

    # Threshold stability
    stab_thresh, stab_std = threshold_stability(df_legacy, target_metric='F1', window=10)

    # Latency
    lat_mean = np.mean(latency_ms_list) if latency_ms_list else 0
    lat_p99  = np.percentile(latency_ms_list, 99) if latency_ms_list else 0

    # Calibration (on confidence scores)
    conf_scores     = [r['confidence'] for r in all_results]
    impostor_binary = [0 if r['true'] == r['top1_label'] else 1 for r in all_results]
    cal_probs       = platt_scale([r['top1_dist'] for r in all_results], impostor_binary)
    try:
        frac_pos, mean_pred = calibration_curve(impostor_binary, cal_probs, n_bins=10)
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])

    # ── K1/K2 grid search ─────────────────────────────────────────────────────
    print("🔍  Running K1/K2 grid search …")
    df_grid = sweep_k1_k2(
        processed_data, person_data, min_vids,
        k2_range  = [5, 7, 9, 11, 15],
        k1_fracs  = [0.5, 0.6, 0.67, 0.75, 0.8]
    )
    df_grid.to_csv(f"{RUN_FOLDER}/k1k2_grid_search.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    DARK   = "#0f1117"
    GRID_C = "#1e2130"
    ACCENT = "#00e5ff"
    GREEN  = "#00ff9f"
    RED    = "#ff4757"
    ORANGE = "#ffa502"
    PURPLE = "#a29bfe"
    TEXT   = "#e8eaf6"
    plt.rcParams.update({
        "figure.facecolor": DARK,   "axes.facecolor": GRID_C,
        "axes.edgecolor":   "#333", "axes.labelcolor": TEXT,
        "xtick.color": TEXT,        "ytick.color": TEXT,
        "text.color":  TEXT,        "grid.color": "#2a2d3e",
        "legend.facecolor": GRID_C, "legend.edgecolor": "#444",
        "font.family": "monospace"
    })

    # ── P1: Embedding Clusters ────────────────────────────────────────────────
    X_vis = np.vstack(all_embs_for_vis)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor=DARK)
    fig.suptitle("Embedding Space Projections", color=TEXT, fontsize=16, fontweight='bold')
    unique_labels   = sorted(set(all_labels_for_vis))
    palette         = sns.color_palette("tab10", n_colors=len(unique_labels))
    label_to_color  = {l: palette[i] for i, l in enumerate(unique_labels)}
    colors          = [label_to_color[l] for l in all_labels_for_vis]
    pca_res         = PCA(n_components=2).fit_transform(X_vis)
    axes[0].scatter(pca_res[:,0], pca_res[:,1], c=colors, s=30, alpha=0.7, edgecolors='none')
    axes[0].set_title("PCA Projection", color=TEXT); axes[0].grid(True, alpha=0.2)
    safe_perp = min(30, max(1, X_vis.shape[0] - 1))
    try:
        tsne_res = TSNE(n_components=2, perplexity=safe_perp, init='pca',
                        learning_rate='auto').fit_transform(X_vis)
        axes[1].scatter(tsne_res[:,0], tsne_res[:,1], c=colors, s=30, alpha=0.7, edgecolors='none')
        axes[1].set_title(f"t-SNE (perp={safe_perp})", color=TEXT)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"t-SNE Error:\n{e}", ha='center', va='center',
                     transform=axes[1].transAxes)
    axes[1].grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/clusters_projection.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── P2: ROC + AUC ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0,1],[0,1], 'k--', alpha=0.4, label="Chance")
    eer_fpr_idx = np.argmin(np.abs(fpr - eer))
    ax.scatter([fpr[eer_fpr_idx]], [tpr[eer_fpr_idx]], color=GREEN, s=80, zorder=5,
               label=f"EER ≈ {eer:.4f}")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1-FRR)")
    ax.set_title(f"ROC  |  AUC={roc_auc:.4f}  |  EER={eer:.4f}", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/roc_auc.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P3: DET Curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=DARK)
    ax.plot(det_fpr * 100, det_fnr * 100, color=RED, lw=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    diag = np.linspace(1e-3, 100, 300)
    ax.plot(diag, diag, 'k--', alpha=0.3, label="EER line")
    ax.axvline(eer*100, color=GREEN, linestyle=':', label=f"EER={eer*100:.2f}%")
    ax.axhline(eer*100, color=GREEN, linestyle=':', alpha=0.6)
    ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)")
    ax.set_title(f"DET Curve  |  EER≈{eer*100:.2f}%", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2, which='both')
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/det_curve.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P4: Score Distributions ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=DARK)
    sns.kdeplot(same_dist, ax=axes[0], label="Genuine",  fill=True, color=GREEN, alpha=0.5)
    sns.kdeplot(diff_dist, ax=axes[0], label="Impostor", fill=True, color=RED,   alpha=0.5)
    axes[0].axvline(PRODUCTION_THRESHOLD, color='white', linestyle='--',
                    label=f"Legacy T={PRODUCTION_THRESHOLD}")
    axes[0].axvline(DIST_GATE, color=ORANGE, linestyle=':',
                    label=f"Consensus dist_gate={DIST_GATE}")
    axes[0].axvline(-eer_threshold, color=PURPLE, linestyle='-.',
                    label=f"EER T≈{-eer_threshold:.3f}")
    axes[0].set_title(f"Score Distributions  |  Sep Ratio={sep_ratio:.3f}", fontweight='bold')
    axes[0].set_xlabel("L2 Distance"); axes[0].legend(); axes[0].grid(True, alpha=0.2)
    axes[1].hist(same_dist, bins=40, color=GREEN, alpha=0.6, label="Genuine",  density=True)
    axes[1].hist(diff_dist, bins=40, color=RED,   alpha=0.6, label="Impostor", density=True)
    axes[1].set_title("Score Histograms", fontweight='bold')
    axes[1].set_xlabel("L2 Distance"); axes[1].legend(); axes[1].grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/score_distributions.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P5: FAR/FRR (legacy threshold) + operating points ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK)
    ax.plot(df_legacy['Threshold'], df_legacy['FAR'], color=RED,   lw=2, label='FAR (legacy)')
    ax.plot(df_legacy['Threshold'], df_legacy['FRR'], color=ACCENT, lw=2, label='FRR (legacy)')
    ax.axvline(PRODUCTION_THRESHOLD, color='white',  linestyle='--',
               label=f"Prod T={PRODUCTION_THRESHOLD}")
    ax.axvline(-eer_threshold, color=PURPLE, linestyle=':',
               label=f"EER T≈{-eer_threshold:.3f}")
    # Consensus operating points as horizontal lines
    ax.axhline(consensus_far, color=GREEN,  linestyle='-.', lw=1.5,
               label=f"Consensus FAR={consensus_far:.4f}")
    ax.axhline(consensus_frr, color=ORANGE, linestyle='-.', lw=1.5,
               label=f"Consensus FRR={consensus_frr:.4f}")
    if op_row is not None:
        ax.axvline(op_row['Threshold'], color=GREEN, linestyle='--', alpha=0.5,
                   label=f"Op.Point T={op_row['Threshold']:.3f}")
    ax.set_title("FAR vs FRR — Legacy vs Consensus", fontweight='bold')
    ax.set_xlabel("Threshold (legacy)"); ax.set_ylabel("Error Rate")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/security_analysis.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P6: Consensus Confusion Matrix ───────────────────────────────────────
    labels_cm = sorted(person_data.keys()) + ["unknown"]
    cm = confusion_matrix(y_true_c, y_pred_c, labels=labels_cm)
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=DARK)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_cm, yticklabels=labels_cm[:-1], ax=ax)
    ax.set_title(f"Consensus Confusion Matrix  (K2={K2}, K1={K1}, gate={DIST_GATE})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/confusion_matrix.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P7: K1/K2 Grid Search Heatmap ────────────────────────────────────────
    for metric in ['FAR', 'FRR', 'F1']:
        pivot = df_grid.pivot(index='K1_frac', columns='K2', values=metric)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK)
        cmap = 'RdYlGn' if metric == 'F1' else 'RdYlGn_r'
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                    linewidths=0.5, linecolor='#333')
        ax.set_title(f"K1/K2 Grid Search — {metric}", fontweight='bold')
        ax.set_xlabel("K2 (neighbourhood size)")
        ax.set_ylabel("K1 fraction (consensus threshold)")
        plt.tight_layout()
        plt.savefig(f"{RUN_FOLDER}/k1k2_grid_{metric.lower()}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ── P8: Vote Confidence Distribution ─────────────────────────────────────
    correct_conf  = [r['confidence'] for r in all_results if r['pred'] == r['true'] and r['pred'] != 'unknown']
    wrong_conf    = [r['confidence'] for r in all_results if r['pred'] != r['true'] and r['pred'] != 'unknown']
    unknown_conf  = [r['confidence'] for r in all_results if r['pred'] == 'unknown']
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    if correct_conf:  sns.kdeplot(correct_conf,  ax=ax, label="Correct accept",   fill=True, color=GREEN,  alpha=0.5)
    if wrong_conf:    sns.kdeplot(wrong_conf,     ax=ax, label="Wrong accept",     fill=True, color=RED,    alpha=0.5)
    if unknown_conf:  sns.kdeplot(unknown_conf,   ax=ax, label="Rejected (unknown)", fill=True, color=ORANGE, alpha=0.5)
    ax.set_xlabel("Vote Confidence (weighted share of winning class)")
    ax.set_title("Confidence Distribution by Decision Type", fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/confidence_distribution.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P9: Calibration ───────────────────────────────────────────────────────
    if len(frac_pos):
        fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK)
        ax.plot(mean_pred, frac_pos, 's-', color=ACCENT, lw=2, label="Platt-calibrated")
        ax.plot([0,1],[0,1], 'k--', alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Predicted probability (impostor)")
        ax.set_ylabel("Fraction of true impostors")
        ax.set_title("Calibration Curve (Platt Scaling)", fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{RUN_FOLDER}/calibration_curve.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P10: Latency ──────────────────────────────────────────────────────────
    if latency_ms_list:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK)
        ax.hist(latency_ms_list, bins=40, color=ACCENT, alpha=0.8, edgecolor='none')
        ax.axvline(lat_mean, color=GREEN, linestyle='--', label=f"Mean={lat_mean:.2f}ms")
        ax.axvline(lat_p99,  color=RED,   linestyle=':',  label=f"P99={lat_p99:.2f}ms")
        ax.set_xlabel("Inference Latency (ms)"); ax.set_ylabel("Count")
        ax.set_title("Inference Latency Distribution", fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{RUN_FOLDER}/latency_distribution.png", dpi=150, bbox_inches='tight'); plt.close()

    # ── P11: Threshold Stability ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    ax.plot(df_legacy['Threshold'], df_legacy['F1'],       color=ACCENT, lw=2,   label="F1")
    ax.plot(df_legacy['Threshold'], df_legacy['Accuracy'], color=GREEN,  lw=1.5, linestyle='--', label="Accuracy")
    ax.axvspan(stab_thresh - stab_std, stab_thresh + stab_std,
               alpha=0.15, color=ORANGE, label=f"Stability band (σ={stab_std:.4f})")
    ax.axvline(stab_thresh, color=ORANGE, linestyle=':', label=f"Best T zone≈{stab_thresh:.3f}")
    ax.axvline(PRODUCTION_THRESHOLD, color='white', linestyle='--', label=f"Prod T={PRODUCTION_THRESHOLD}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Threshold Stability  σ={stab_std:.4f} → "
                 f"{'STABLE' if stab_std < 0.05 else 'UNSTABLE'}", fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{RUN_FOLDER}/threshold_stability.png", dpi=150, bbox_inches='tight'); plt.close()

    # ═══════════════════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    best_legacy = df_legacy.iloc[(df_legacy['Threshold'] - PRODUCTION_THRESHOLD).abs().argsort().iloc[0]]

    report_lines = [
        "=" * 65,
        f"  FULL EVALUATION REPORT (Open-Set K1/K2)  |  {RUN_FOLDER}",
        "=" * 65,
        "",
        "── Core Biometric Metrics (ROC/AUC/EER on top-1 distances) ──",
        f"  ROC AUC              : {roc_auc:.4f}",
        f"  EER                  : {eer*100:.2f}%  (at dist≈{-eer_threshold:.4f})",
        f"  Separation Ratio     : {sep_ratio:.3f}",
        "",
        f"── Legacy Threshold System  (T={PRODUCTION_THRESHOLD}) ──",
        f"  Accuracy             : {float(best_legacy['Accuracy']):.4f}",
        f"  FAR                  : {float(best_legacy['FAR']):.4f}",
        f"  FRR                  : {float(best_legacy['FRR']):.4f}",
        f"  F1                   : {float(best_legacy['F1']):.4f}",
        "",
        f"── Open-Set Consensus  (K2={K2}, K1={K1}, dist_gate={DIST_GATE}) ──",
        f"  FAR                  : {consensus_far:.4f}  ({'✅ better' if consensus_far < float(best_legacy['FAR']) else '⚠️ worse'} than legacy)",
        f"  FRR                  : {consensus_frr:.4f}  ({'✅ better' if consensus_frr < float(best_legacy['FRR']) else '⚠️ worse'} than legacy)",
        f"  Accuracy             : {consensus_acc:.4f}",
        f"  F1                   : {consensus_f1:.4f}",
        f"  Unknown Rate         : {unknown_rate*100:.1f}%  (samples rejected as unknown)",
        "",
        "── Operating Points ──",
    ]
    if op_row is not None:
        report_lines.append(
            f"  FAR≤{FAR_CONSTRAINT} operating point : T={op_row['Threshold']:.4f}, "
            f"FAR={op_row['FAR']:.4f}, FRR={op_row['FRR']:.4f}"
        )
    else:
        report_lines.append(f"  FAR≤{FAR_CONSTRAINT} operating point : not achievable in sweep")

    report_lines += [
        "",
        "── Operational Metrics ──",
        f"  Inference latency    : mean={lat_mean:.2f}ms, p99={lat_p99:.2f}ms",
        f"  Threshold stability  : best_zone≈{stab_thresh:.4f}, σ={stab_std:.4f}",
        "",
        "── Output Files ──",
        "  clusters_projection.png     roc_auc.png",
        "  det_curve.png               score_distributions.png",
        "  security_analysis.png       confusion_matrix.png",
        "  confidence_distribution.png calibration_curve.png",
        "  latency_distribution.png    threshold_stability.png",
        "  k1k2_grid_far.png           k1k2_grid_frr.png",
        "  k1k2_grid_f1.png            k1k2_grid_search.csv",
        "  legacy_threshold_sweep.csv  summary.csv",
        "=" * 65,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open(f"{RUN_FOLDER}/report.txt", "w") as f:
        f.write(report_text)

    # Summary CSV
    summary = {
        "roc_auc": roc_auc, "eer": eer, "eer_threshold": -eer_threshold,
        "separation_ratio": sep_ratio,
        "legacy_threshold": PRODUCTION_THRESHOLD,
        "legacy_accuracy": float(best_legacy['Accuracy']),
        "legacy_far": float(best_legacy['FAR']),
        "legacy_frr": float(best_legacy['FRR']),
        "legacy_f1":  float(best_legacy['F1']),
        "consensus_k2": K2, "consensus_k1": K1, "consensus_dist_gate": DIST_GATE,
        "consensus_far": consensus_far, "consensus_frr": consensus_frr,
        "consensus_accuracy": consensus_acc, "consensus_f1": consensus_f1,
        "unknown_rate": unknown_rate,
        "lat_mean_ms": lat_mean, "lat_p99_ms": lat_p99,
        "stability_threshold": stab_thresh, "stability_std": stab_std,
    }
    if op_row is not None:
        summary.update({"op_threshold": float(op_row['Threshold']),
                         "op_far": float(op_row['FAR']), "op_frr": float(op_row['FRR'])})

    pd.DataFrame([summary]).to_csv(f"{RUN_FOLDER}/summary.csv", index=False)
    print(f"\n✅  Evaluation complete → {RUN_FOLDER}/")


if __name__ == "__main__":
    run_full_evaluation()

(venv_voice311) antonio@antonio-IdeaPad-Slim-3-15IAH8:~/Desktop/NMDU$ python3 eval_embeddings.py
🚀  Full evaluation (open-set K1/K2) → full_eval_report_9
✅  Model loaded: triplet_run_2/best_triplet_model.pth
🧬 Extraction: 100%|████████████████████████████| 40/40 [00:56<00:00,  1.41s/it]
🔍  Running K1/K2 grid search …

=================================================================
  FULL EVALUATION REPORT (Open-Set K1/K2)  |  full_eval_report_9
=================================================================

── Core Biometric Metrics (ROC/AUC/EER on top-1 distances) ──
  ROC AUC              : 0.8652
  EER                  : 21.61%  (at dist≈0.0650)
  Separation Ratio     : 0.655

── Legacy Threshold System  (T=0.1) ──
  Accuracy             : 0.9014
  FAR                  : 0.3929
  FRR                  : 0.0883
  F1                   : 0.9420

── Open-Set Consensus  (K2=9, K1=6, dist_gate=0.35) ──
  FAR                  : 0.5714  (⚠️ worse than legacy)
  FRR                  : 0.0049  (✅ better than legacy)
  Accuracy             : 0.9859
  F1                   : 0.9906
  Unknown Rate         : 1.0%  (samples rejected as unknown)

── Operating Points ──
  FAR≤0.01 operating point : T=0.0398, FAR=0.0000, FRR=0.4857

── Operational Metrics ──
  Inference latency    : mean=6.66ms, p99=7.72ms
  Threshold stability  : best_zone≈0.8582, σ=0.0151

── Output Files ──
  clusters_projection.png     roc_auc.png
  det_curve.png               score_distributions.png
  security_analysis.png       confusion_matrix.png
  confidence_distribution.png calibration_curve.png
  latency_distribution.png    threshold_stability.png
  k1k2_grid_far.png           k1k2_grid_frr.png
  k1k2_grid_f1.png            k1k2_grid_search.csv
  legacy_threshold_sweep.csv  summary.csv
=================================================================

✅  Evaluation complete → full_eval_report_9/


