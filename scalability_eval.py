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
    base = "scalability_tests_"
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

def run_scalability_test(processed_data, RUN_FOLDER):
    print("\n📈 Pokrećem Scalability Test (dodavanje klase po klasu)...")
    all_p_ids = sorted(list(processed_data.keys()))
    scalability_results = []
    output_file = os.path.join(RUN_FOLDER, "scalability_report.csv")
    
    for n in range(1, len(all_p_ids) + 1):
        current_p_ids = all_p_ids[:n]
        current_data = {p: processed_data[p] for p in current_p_ids}
        
        fold_results = []
        # Pretpostavljamo da sve osobe imaju bar min_vids foldova
        min_vids = len(next(iter(current_data.values())))
        
        for fold in range(min_vids):
            gallery_embs, gallery_labels = [], []
            query_embs, query_labels = [], []
            
            for p_id, vids_embs in current_data.items():
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
                dist = float(D[i][0])
                # Predikcija uzimajući u obzir prag
                pred = gallery_labels[I[i][0]] if dist < PRODUCTION_THRESHOLD else "unknown"
                fold_results.append({'true': query_labels[i], 'pred': pred})

        y_true = [r['true'] for r in fold_results]
        y_pred = [r['pred'] for r in fold_results]
        
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        tp = sum(1 for t, pv in zip(y_true, y_pred) if t == pv)
        fp = sum(1 for t, pv in zip(y_true, y_pred) if t != pv and pv != "unknown")
        fn = sum(1 for t, pv in zip(y_true, y_pred) if pv == "unknown")

        scalability_results.append({
            'num_classes': n, 'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn
        })
        print(f" > Klasa: {n:02d} | Acc: {acc:.4f} | F1: {f1:.4f}")

    df_scal = pd.DataFrame(scalability_results)
    df_scal.to_csv(output_file, index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_scal['num_classes'], df_scal['accuracy'], 'o-', label='Accuracy')
    plt.plot(df_scal['num_classes'], df_scal['f1'], 's-', label='F1 Score')
    plt.xlabel('Broj osoba u sustavu')
    plt.ylabel('Score')
    plt.title('Pad performansi s povećanjem broja korisnika')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RUN_FOLDER, "scalability_graph.png"))
    plt.close()

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
    all_embs_for_vis, all_labels_for_vis = [], []

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

    # --- 1. TEST SKALABILNOSTI ---
    run_scalability_test(processed_data, RUN_FOLDER)

    # --- 2. PUNAL EVALUACIJA (Cross-validation) ---
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
                'true': query_labels[i], 'pred_raw': gallery_labels[I[i][0]], 'dist': float(D[i][0])
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
        sweep_data.append({'Threshold': t, 'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1, 'FAR': far, 'FRR': frr})

    df = pd.DataFrame(sweep_data)
    df.to_csv(f"{RUN_FOLDER}/metrics_sweep.csv", index=False)

    # --- VIZUALIZACIJE ---
    X_vis = np.vstack(all_embs_for_vis)
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    pca_res = PCA(n_components=2).fit_transform(X_vis)
    sns.scatterplot(x=pca_res[:,0], y=pca_res[:,1], hue=all_labels_for_vis, palette='tab10', s=40, alpha=0.7, legend=False)
    plt.title("PCA Projekcija")

    plt.subplot(1, 2, 2)
    tsne = TSNE(n_components=2, perplexity=min(30, X_vis.shape[0]-1), init='pca', learning_rate='auto')
    tsne_res = tsne.fit_transform(X_vis)
    sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=all_labels_for_vis, palette='tab10', s=40, alpha=0.7, legend=False)
    plt.title("t-SNE Projekcija")
    plt.savefig(f"{RUN_FOLDER}/clusters_projection.png")
    plt.close()

    # --- SECURITY DISTRIBUTIONS ---
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    sns.kdeplot(same_dist, label="Same Person", fill=True, color="green")
    sns.kdeplot(diff_dist, label="Different Person", fill=True, color="red")
    plt.axvline(PRODUCTION_THRESHOLD, color='black', linestyle='--')
    plt.title("Distribucija distanci")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['Threshold'], df['FAR'], label='FAR', color='red')
    plt.plot(df['Threshold'], df['FRR'], label='FRR', color='blue')
    plt.axvline(PRODUCTION_THRESHOLD, color='black', linestyle='--')
    plt.title("FAR vs FRR")
    plt.legend()
    plt.savefig(f"{RUN_FOLDER}/security_analysis.png")
    plt.close()

    # --- CONFUSION MATRIX ---
    y_true_s = [res['true'] for res in all_results]
    y_pred_s = [res['pred_raw'] if res['dist'] < PRODUCTION_THRESHOLD else "unknown" for res in all_results]
    labels = sorted(list(person_data.keys())) + ["unknown"]
    cm = confusion_matrix(y_true_s, y_pred_s, labels=labels)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels[:-1])
    plt.title(f"Confusion Matrix (T={PRODUCTION_THRESHOLD})")
    plt.savefig(f"{RUN_FOLDER}/confusion_matrix_strict.png")
    plt.close()

    # FINAL REPORT
    best_row = df.iloc[(df['Threshold'] - PRODUCTION_THRESHOLD).abs().argsort()[:1]]
    print("\n" + "="*50)
    print(f"📊 FINALNI REPORT ZA PRAG: {PRODUCTION_THRESHOLD}")
    print("="*50)
    print(best_row[['Threshold', 'Accuracy', 'FAR', 'FRR']].to_string(index=False))
    print(f"✅ Svi rezultati spremljeni u: {RUN_FOLDER}")

if __name__ == "__main__":
    run_full_evaluation()

'''

num_classes,accuracy,precision,recall,f1,tp,fp,fn
1,0.9703703703703703,1.0,0.9703703703703703,0.9849624060150377,131,0,4
2,0.9116022099447514,1.0,0.9116022099447514,0.9506625680222657,165,0,16
3,0.8590909090909091,1.0,0.8590909090909091,0.9172009569377991,189,0,31
4,0.8913738019169329,1.0,0.8913738019169329,0.9369316944485028,279,0,34
5,0.8474114441416893,1.0,0.8474114441416893,0.9085712993842215,311,0,56
6,0.8391608391608392,1.0,0.8391608391608392,0.90485875434207,360,0,69
7,0.8515769944341373,1.0,0.8515769944341373,0.9135341965278989,459,0,80
8,0.8498349834983498,1.0,0.8498349834983498,0.9132063729535286,515,0,91
9,0.8716502115655853,1.0,0.8716502115655853,0.9258153201831288,618,0,91
10,0.8736842105263158,1.0,0.8736842105263158,0.9273344684611274,664,0,96
11,0.8550185873605948,1.0,0.8550185873605948,0.9148124511589236,690,0,117
12,0.8597914252607184,1.0,0.8597914252607184,0.9179369339642842,742,0,121
13,0.8611422172452408,1.0,0.8611422172452408,0.9189256737320898,769,0,124
14,0.8632478632478633,1.0,0.8632478632478633,0.9204092581913957,808,0,128
15,0.8732943469785575,1.0,0.8732943469785575,0.926405292117047,896,0,130
16,0.8746518105849582,0.997051339324819,0.8746518105849582,0.9260852515646959,942,3,132
17,0.880349344978166,0.9972264562906814,0.880349344978166,0.929588550709208,1008,3,134
18,0.8740490278951818,0.997315547297405,0.8740490278951818,0.9258274645494871,1034,3,146
19,0.8730675345809601,0.9974160231512044,0.8730675345809601,0.9255212817477018,1073,3,153
20,0.8778386844166014,0.9975131499239077,0.8778386844166014,0.9283207950414452,1121,3,153
21,0.8799403430275914,0.9976318362810068,0.8799403430275914,0.9298016623800637,1180,3,158
22,0.8784570596797671,0.9976887135755678,0.8784570596797671,0.9290859019298875,1207,3,164
23,0.8847203274215553,0.9978337601997477,0.8847203274215553,0.932846548595277,1297,3,166
24,0.8856015779092702,0.9979120923424262,0.8856015779092702,0.9335529208557715,1347,3,171
25,0.8878326996197718,0.9979875110600952,0.8878326996197718,0.9349768390881419,1401,3,174
26,0.8877300613496932,0.9980517131612455,0.8877300613496932,0.9350980236881596,1447,3,180
27,0.8833836858006042,0.9980811434760303,0.8833836858006042,0.9323019810342599,1462,3,190
28,0.8816254416961131,0.9981297364268729,0.8816254416961131,0.9314190354543249,1497,3,198
29,0.8851428571428571,0.9981853099730458,0.8851428571428571,0.9334568698293965,1549,3,198
30,0.8832402234636871,0.9958828613768687,0.8832402234636871,0.9314281569818053,1581,7,202
31,0.8913934426229508,0.9957064989290669,0.8913934426229508,0.9360887491663068,1740,8,204
32,0.8909547738693467,0.9957884853816777,0.8909547738693467,0.9359644130827149,1773,8,209
33,0.8926336061627347,0.9954659230973399,0.8926336061627347,0.9369123088263294,1854,9,214
34,0.8962350780532599,0.9956761810253328,0.8962350780532599,0.9391387751914415,1952,9,217
35,0.8951142985208427,0.9957788983743501,0.8951142985208427,0.9386453257528977,1997,9,225
36,0.8966578715919086,0.9958587169187225,0.8966578715919086,0.939583042832794,2039,9,226
37,0.8976784178847808,0.9959512993435834,0.8976784178847808,0.940269683983804,2088,9,229
38,0.9002087682672234,0.9960679424940188,0.9002087682672234,0.9417802235120214,2156,9,230
39,0.9007383100902379,0.995335593444885,0.9007383100902379,0.9417067925256318,2196,11,231
40,0.9021827000808408,0.9954034667819844,0.9021827000808408,0.942555036450077,2232,11,231

'''
