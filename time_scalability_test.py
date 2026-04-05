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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

# --- TEST 1: TIME SCALABILITY (LATENCY) ---
def run_time_scalability_test(processed_data, model, RUN_FOLDER):
    print("\n⏱️ TEST 1/3: Vremenska skalabilnost (Latency)...")
    all_p_ids = sorted(list(processed_data.keys()))
    time_results = []
    
    dummy_input = torch.randn(1, 1, 128, 128).to(DEVICE)
    with torch.inference_mode():
        for _ in range(20): _ = model(dummy_input) # Warmup
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
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='weighted', zero_division=0)
        
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

    # 1. LOAD MODEL
    model = VoiceNetEmbedding(EMBEDDING_DIM).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model ne postoji na {MODEL_PATH}"); return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. DATA EXTRACTION
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

    # 3. TESTS
    run_time_scalability_test(processed_data, model, RUN_FOLDER)
    run_accuracy_scalability_test(processed_data, RUN_FOLDER)

    # 4. FINAL CROSS-VAL & PLOTS (Full Dataset)
    print("\n🎨 Generiram finalne vizualizacije...")
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

    # Security Analysis
    same_d = [r['dist'] for r in all_results if r['true'] == r['pred_raw']]
    diff_d = [r['dist'] for r in all_results if r['true'] != r['pred_raw']]
    plt.figure(figsize=(10, 5))
    sns.kdeplot(same_d, label="Same Person", fill=True, color="green")
    sns.kdeplot(diff_d, label="Different Person", fill=True, color="red")
    plt.axvline(PRODUCTION_THRESHOLD, color='black', ls='--')
    plt.title("Distribucija distanci"); plt.legend(); plt.savefig(f"{RUN_FOLDER}/distance_dist.png"); plt.close()

    # Projection (t-SNE)
    X_proj = np.vstack(all_embs_flat)
    tsne_res = TSNE(n_components=2, perplexity=min(30, len(X_proj)-1)).fit_transform(X_proj)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=all_labs_flat, palette='tab20', s=30, alpha=0.6, legend=False)
    plt.title("t-SNE Klasteri (svaka boja je jedna osoba)"); plt.savefig(f"{RUN_FOLDER}/tsne_clusters.png"); plt.close()

    # Confusion Matrix (Final)
    y_true_f = [r['true'] for r in all_results]
    y_pred_f = [r['pred_raw'] if r['dist'] < PRODUCTION_THRESHOLD else "unknown" for r in all_results]
    labels = sorted(list(processed_data.keys())) + ["unknown"]
    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels[:-1])
    plt.title("Confusion Matrix Heatmap"); plt.savefig(f"{RUN_FOLDER}/confusion_matrix.png"); plt.close()

    print(f"\n✅ SVE GOTOVO! Izvještaj je spreman u folderu: {RUN_FOLDER}")

if __name__ == "__main__":
    run_full_evaluation()

num_classes,num_vectors,inference_ms,indexing_ms,search_ms,total_latency_ms
1,131,7.415885340014938,0.1468589762225747,0.016172099858522415,7.432057439873462
2,183,7.415885340014938,0.04367198562249541,0.016993310127872974,7.432878650142811
3,227,7.415885340014938,0.044002983486279845,0.018558660231065005,7.434444000246004
4,455,7.415885340014938,0.05666300421580672,0.02655516000231728,7.4424405000172555
5,506,7.415885340014938,0.06125099025666714,0.02828511002007872,7.444170450035017
6,568,7.415885340014938,0.07363501936197281,0.030499499989673495,7.446384840004613
7,739,7.415885340014938,0.12408499605953693,0.03808210021816194,7.4539674402331
8,800,7.415885340014938,0.13880699407309294,0.03862954006763175,7.45451488008257
9,956,7.415885340014938,0.1235610106959939,0.03748368006199598,7.453369020076934
10,1018,7.415885340014938,0.11217599967494607,0.02364446991123259,7.439529809926172
11,1067,7.415885340014938,0.06895698606967926,0.02448154002195224,7.4403668800368905
12,1141,7.415885340014938,0.07897202158346772,0.02589531010016799,7.441780650115107
13,1223,7.415885340014938,0.07556902710348368,0.02752512024017051,7.44341046025511
14,1452,7.415885340014938,0.10014697909355164,0.03243412997107953,7.448319469986018
15,1499,7.415885340014938,0.09278999641537666,0.03196209989255294,7.447847439907491
16,1591,7.415885340014938,0.09881900041364133,0.033977500279434025,7.449862840294373
17,1739,7.415885340014938,0.13494500308297575,0.03709356009494513,7.452978900109884
18,1782,7.415885340014938,0.1407009840477258,0.03831717011053115,7.454202510125469
19,1817,7.415885340014938,0.12304799747653306,0.038645890017505735,7.454531230032444
20,1903,7.415885340014938,0.13344400213100016,0.04016569000668824,7.4560510300216265
21,2036,7.415885340014938,0.1312730018980801,0.042694889998529106,7.458580230013467
22,2111,7.415885340014938,0.1772760006133467,0.05290797998895869,7.468793320003897
23,2370,7.415885340014938,0.16855399007909,0.04844334005611017,7.464328680071049
24,2560,7.415885340014938,0.16993199824355543,0.05250604008324444,7.468391380098184
25,2684,7.415885340014938,0.17321101040579379,0.05528443987714127,7.4711697798920795
26,2771,7.415885340014938,0.18242700025439262,0.05719578999560326,7.4730811300105415
27,2866,7.415885340014938,0.18825600272975862,0.05904627003474161,7.47493161004968
28,2939,7.415885340014938,0.19645600696094334,0.060770990094169974,7.476656330109108
29,3065,7.415885340014938,0.20363600924611092,0.06057466001948342,7.476460000034422
30,3112,7.415885340014938,0.203340983716771,0.061002020083833486,7.476887360098772
31,3380,7.415885340014938,0.2452520129736513,0.06784237979445606,7.483727719809394
32,3427,7.415885340014938,0.24076300906017423,0.0785438102320768,7.494429150247016
33,3562,7.415885340014938,0.2322870132047683,0.08327557006850839,7.4991609100834475
34,3664,7.415885340014938,0.34262999542988837,0.06867425021482632,7.4845595902297655
35,3817,7.415885340014938,0.323473010212183,0.06961648992728442,7.485501829942223
36,3942,7.415885340014938,0.2571299846749753,0.07069050014251843,7.486575840157458
37,4125,7.415885340014938,0.3256170020904392,0.07293375005247071,7.488819090067409
38,4260,7.415885340014938,0.3323450218886137,0.07618325995281339,7.492068599967752
39,4318,7.415885340014938,0.3213740128558129,0.07637650007382035,7.492261840088759
40,4472,7.415885340014938,0.3463079920038581,0.07953406020533293,7.495419400220271


num_classes,accuracy,f1
1,0.9746835443037974,0.9871794871794872
2,0.9138755980861244,0.9514867110560892
3,0.8532818532818532,0.9125896625896627
4,0.8864265927977839,0.933070178811693
5,0.8481927710843373,0.908492484488311
6,0.8378378378378378,0.9034570532247622
7,0.8594507269789984,0.9174655205296481
8,0.8571428571428571,0.9167919639783572
9,0.8765743073047859,0.9281099336135428
10,0.8788235294117647,0.9297676057167605
11,0.8680479825517994,0.923561519231307
12,0.8716904276985743,0.9259734349644689
13,0.8756121449559255,0.9283050143277327
14,0.8802946593001841,0.9311823080642255
15,0.8877551020408163,0.9355890736417455
16,0.8902340597255851,0.936403954533371
17,0.8974358974358975,0.9405765457517697
18,0.8890510948905109,0.9353479721818021
19,0.8871650211565585,0.9344591576342967
20,0.8910823689584751,0.9367345714945082
21,0.8927648578811369,0.9379487496594321
22,0.892432770481551,0.9379344368185122
23,0.8979351032448377,0.9408596840547496
24,0.9007332205301748,0.9425988452950338
25,0.903418339663592,0.9442285729541354
26,0.9036842105263158,0.9442653521291478
27,0.9008776458440888,0.9426851275799194
28,0.8990918264379415,0.9417713664076007
29,0.9019607843137255,0.9434268863822867
30,0.8999521302058401,0.9416818409816091
31,0.9067017082785808,0.945752313831471
32,0.9047619047619048,0.944739174275362
33,0.9057676685621446,0.9453664922518981
34,0.9083820662768031,0.9469669147900539
35,0.9063205417607223,0.945974555678942
36,0.9077883908890522,0.9466894041039696
37,0.9093839541547278,0.9476626308085181
38,0.9114056505057552,0.9488561944585575
39,0.9116135662898253,0.9486649368028378
40,0.912559081701553,0.9492393883101088

