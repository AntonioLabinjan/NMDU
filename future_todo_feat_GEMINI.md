Evo ti sažeti **Future Feature TODO** koji možeš spremiti u `README.md` ili u bilješke, napisan tako da ti točno kaže što trebaš promijeniti u kodu.

---

### 🚀 Future Feature: Siamese Network & Embedding Search (Open-Set Recognition)

#### 1. Zašto (The "Why")

* **Open-Set problem:** Trenutni model mora znati sve ljude unaprijed. Siamese model prepoznaje "nepoznate" jer njihovi vektori ne odgovaraju nikome u bazi.
* **Zero-shot dodavanje:** Dodavanje nove osobe u sustav traje 0.1s (samo jedan "prolaz" kroz mrežu), bez ponovnog treninga cijele mreže.
* **Skalabilnost:** Pomoću FAISS-a možeš pretraživati bazu od milijun glasova brže nego što trenutni model klasificira njih 40.

#### 2. Tehnička Implementacija (The "How")

**A. Promjena Arhitekture (Encoder)**
Izbaci zadnji `Linear` sloj iz svog `VoiceNetMedium` modela. Umjesto da vraća vjerojatnosti za klase (npr. 40 brojeva), model će vraćati **Embedding** (niz od npr. 128 ili 256 brojeva).

**B. Nova strategija treninga (Triplet Loss)**
Umjesto da šalješ jedan po jedan audio, loader mora generirati **triplete**:

1. **Anchor (A):** Isječak glasa osobe 1.
2. **Positive (P):** Drugi isječak iste osobe 1.
3. **Negative (N):** Isječak osobe 2 (bilo koga drugoga).
*Loss funkcija* (`nn.TripletMarginLoss`) će tjerati model da smanji udaljenost (A, P) i poveća udaljenost (A, N).

**C. Baza podataka (FAISS)**
Nakon treninga, napravit ćeš skriptu `enroll_users.py`:

* Provjeriš po jedan audio za svakog poznatog korisnika.
* Spremiš dobivene vektore u **FAISS index**.
* FAISS radi "Nearest Neighbor" pretragu – traži točku koja je najbliža onoj koju mu pošalješ.

#### 3. TODO Lista za implementaciju:

* [ ] **Modifikacija dataseta:** Napraviti `TripletVoiceDataset` koji nasumično bira parove (isti/različiti).
* [ ] **Embedding Head:** Dodati `L2 Normalization` na izlaz modela (tako da svi vektori imaju duljinu 1, što olakšava izračun sličnosti).
* [ ] **Loss Switch:** Zamijeniti `nn.CrossEntropyLoss` sa `nn.TripletMarginLoss`.
* [ ] **Inference Update:** Umjesto `model(x).argmax()`, napraviti:
1. `v = model(x)`
2. `distanca, index = faiss_index.search(v, k=1)`
3. Ako je `distanca > threshold` -> "Nepoznata osoba".



---

### 💡 Tehnički savjet:

Tvoj trenutni trening je zapravo **"Pre-training"** za Siamese mrežu. Kad ovaj klasifikator dođe na visok accuracy (npr. >90%), to znači da je tvoj `features` dio (konvolucije) postao ekspert za ljudski glas.

Tada ćemo samo učitati te težine (`best_model.pth`), maknuti klasifikator i krenuti u Siamese trening. To će trajati puno kraće nego da krećeš od nule!
