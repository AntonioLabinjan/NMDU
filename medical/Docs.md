# Izvještaj o treningu modela
**Klasifikacija krvnih stanica — NMDU_cells.ipynb**  
*Iterativni razvoj modela: od linearnog klasifikatora do rezidualne mreže*

---

## 1. Pregled projekta

Ovaj notebook dokumentira sistematičan, iterativan razvoj modela za klasifikaciju krvnih stanica. Polazišna točka je bio jednostavni linearni klasifikator, a svaki sljedeći korak uvodi jedno specifično poboljšanje temeljeno na analizi prethodnih rezultata. Cilj je bio razumjeti koje arhitekturne i trening odluke zapravo doprinose boljem generaliziranju modela.

### 1.1 Skup podataka

Korišten je Kaggle skup podataka `paultimothymooney/blood-cells` (verzija 6). Skup sadrži mikroskopske slike krvnih stanica svrstane u 4 klase:

- EOSINOPHIL
- LYMPHOCYTE
- MONOCYTE
- NEUTROPHIL

Slike su podijeljene na TRAIN i TEST direktorije. Svaka slika je RGB fotografija mikroskopskog uzorka. Za trening je korišten format 64×64 piksela (osim kod transfer learninga gdje je korišten 224×224), batch size 32, a random seed je fiksiran na 42 za reproducibilnost.

### 1.2 Majority baseline

Prije treniranja ikojeg modela izračunata je majority baseline accuracy — točnost koja se postiže uvijek predviđanjem najčešće klase iz trening skupa. Ta vrijednost iznosi **25.09%**. Svaki model koji ne premašuje ovaj prag zapravo ne uči ništa korisno.

### 1.3 Praćenje treninga

U svim eksperimentima trening je praćen na isti način: bilježi se train loss, train accuracy, test loss i test accuracy po epohi. Rezultati se vizualiziraju grafovima gubitka i točnosti. U naprednijim eksperimentima dodano je čuvanje najboljeg modela (best model state) i early stopping s patience parametrom. U finalnom GAP CNN eksperimentu dodano je i čisto evaluiranje trening skupa bez augmentacije (`train_eval`), čime su krivulje učenja postale izravno usporedive.

---

## 2. Eksperimenti — korak po korak

### Korak 00 — Preuzimanje i inspekcija podataka

Prva ćelija preuzima skup podataka s Kagglea korištenjem `kagglehub` biblioteke i ispisuje strukturu direktorija. Provjerava se broj slika po klasi u TRAIN i TEST mapama, te se ispisuje 10 primjera putanja.

**Zaključak:** Skup podataka je uspješno preuzet. Utvrđena je struktura direktorija i distribucija slika po klasama.

---

### Korak 01 — Vizualizacija podataka

Za svaku klasu prikazano je 4 uzoraka iz trening skupa (ukupno 16 slika u gridu 4×4). Ovo je bio ključan korak za razumijevanje vizualnih karakteristika krvnih stanica — svaka klasa ima prepoznatljive morfološke osobine poput oblika jezgre, boje i teksture.

**Zaključak:** Potvrđeno je da se klase vizualno razlikuju, no neke su međusobno slične (osobito EOSINOPHIL i NEUTROPHIL), što je signal da klasifikacija neće biti trivijalna.

---

### Korak 02 — Učitavanje podataka i data loaderi

Postavljeni su PyTorch DataLoaders. Trening transform uključuje samo Resize na 64×64 i ToTensor (bez normalizacije i bez augmentacije). Test transform je identičan. Batch size je 32, shuffle=True za trening.

Klase i njihovi indeksi: `EOSINOPHIL=0, LYMPHOCYTE=1, MONOCYTE=2, NEUTROPHIL=3`

Pikselne vrijednosti su u rasponu [0, 1] (standardni ToTensor). Batch shape: `[32, 3, 64, 64]`.

**Zaključak:** Pipeline za učitavanje podataka je spreman. Podaci nisu normalizirani niti augmentirani — to je namjerno, kao polazna osnova.

---

### Korak 03 — Majority baseline

Izračunata je majority baseline accuracy: **25.09%**. Dominantna klasa je MONOCYTE. Svaki model koji ne premašuje ovu točnost ne uči ništa korisno — radi samo kao prosti gaser.

---

### Run 1 — Linearni klasifikator

#### Arhitektura

Model se sastoji od:
- `Flatten` sloja (pretvara 3×64×64 = 12 288 vrijednosti u vektor)
- jednog `Linear(12288, 4)` sloja — bez skrivenih slojeva i nelinearnosti

#### Hiperparametri

| Parametar | Vrijednost |
|-----------|-----------|
| Epohe | 10 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Input dim | 12 288 (3×64×64) |

#### Praćenje i evaluacija

Nakon svake epohe bilježeni su train loss, train acc, test loss i test acc. Nakon treninga prikazani su grafovi gubitka i točnosti, te confusion matrix i classification report.

#### Zaključak

**Test accuracy: ~29.47%** — jedva bolje od majority baseline (25.09%). Model je gotovo isključivo predviđao MONOCYTE klasu. Linearne granice odluke nisu dovoljne za klasifikaciju slika krvnih stanica jer model ne može modelirati složene vizualne uzorke. Prelazimo na MLP.

---

### Run 2 — MLP s jednim skrivenim slojem (nestabilan)

#### Što je promijenjeno?

Dodan je jedan skriveni sloj s ReLU aktivacijom. Hidden dim = 256. Povećan broj epoha na 15. Ostatak konfiguracije ostao isti.

#### Arhitektura

```
Flatten → Linear(12288, 256) → ReLU → Linear(256, 4)
```

#### Zaključak

**Test accuracy: ~26%** — LOŠIJE od linearnog klasifikatora i gotovo jednako majority baseline! Model je uglavnom predviđao NEUTROPHIL i nije naučio korisne granice odluke za EOSINOPHIL i LYMPHOCYTE. Uzrok: nenormalizirani pikselni ulaz i agresivni learning rate. Trening nije bio stabilan.

---

### Run 3 — Stabilizirani MLP (iste arhitekture, bolji trening setup)

#### Što je promijenjeno?

- Dodana normalizacija ulaza: `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`
- Learning rate smanjen: `1e-3 → 1e-4`
- Optimizer promijenjen: `Adam → AdamW` s `weight_decay=1e-4`
- Broj epoha povećan na 20
- Uveden tracking najboljeg modela (best test acc)

#### Zaključak

**Test accuracy: ~40%** — značajno poboljšanje! Ovo potvrđuje da input normalizacija, manji LR i AdamW s weight decay stabiliziraju optimizaciju. MLP s jednim skrivenim slojem je još uvijek premalen za ovaj problem, ali ispravni trening setup je ključan.

---

### Run 4 — Duboki stabilizirani MLP (BatchNorm + Dropout)

#### Što je promijenjeno?

- Dodan drugi skriveni sloj: `Linear(12288→512) → BN → ReLU → Dropout → Linear(512→256) → BN → ReLU → Dropout → Linear(256→4)`
- `Dropout=0.30`
- Broj epoha: 25

#### Zaključak

**Test accuracy: ~45.2%**, ali train accuracy dostigla ~84% — jasno overfitting. BatchNorm i Dropout poboljšavaju performanse, ali temeljni problem ostaje: MLP flattena sliku i uništava prostornu strukturu. Bez konvolucija model ne može iskoristiti lokalne vizualne uzorke (oblik jezgre, tekstura, granice stanica).

---

### Run 5 — Osnovni CNN

#### Što je promijenjeno?

Potpuni prelazak na konvolucijsku arhitekturu. Ovo je najvažnija tranzicija u projektu.

#### Arhitektura

3 konvolucijska bloka (`Conv2d → ReLU → MaxPool2d`) s filtrima 16/32/64, zatim `Flatten → Linear(4096, 128) → ReLU → Linear(128, 4)`.

#### Hiperparametri

| Parametar | Vrijednost |
|-----------|-----------|
| Epohe | 20 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Normalizacija | Da (mean/std 0.5) |

#### Zaključak

**Test accuracy: ~72.26%** — ogromni skok od ~45%! Konvolucijski slojevi koriste lokalnu prostornu strukturu slike i uče vizualne uzorke karakteristične za svaku vrstu krvne stanice. Ovo jasno pokazuje prednost CNN nad MLP za klasifikaciju slika. Primijećen je ipak i određeni overfitting (train ~88.77% vs test ~72.26%).

---

### Run 6 — Poboljšani CNN (BatchNorm2d + Dropout + augmentacija)

#### Što je promijenjeno?

- Dodana 4. konvolucijska razina (Conv 64→128 kanala)
- `BatchNorm2d` nakon svake konvolucije
- `Dropout(0.40)` u classifier headu
- Data augmentation za trening: `RandomHorizontalFlip`, `RandomRotation(±10°)`, `ColorJitter`
- Early stopping s `patience=7`
- Spremanje najboljeg modela na disk (`best_improved_cnn.pth`)
- LR scheduler: `ReduceLROnPlateau(mode=max, factor=0.5, patience=3)`

#### Zaključak

**Test accuracy: ~79.94%**. Augmentacija pomaže modelu generalizirati na varijacije u rotaciji, osvjetljenju i boji koje su prisutne u mikroskopskim snimkama. Međutim, best accuracy dostignut je vrlo rano (epoch 3), nakon čega train acc raste do ~93% dok test acc stagnira — jasan znak preostalog overfittinga. Glavni klasifikacijski head (Flatten + veliki Linear) potiče memoriranje.

---

### Run 7 — CNN s Global Average Poolingom (GAP)

#### Što je promijenjeno?

Ključna izmjena: zamijenjen `Flatten + Linear` classifier head s **Global Average Poolingom** (`AdaptiveAvgPool2d(1,1) → Flatten → Dropout → Linear(128, 4)`). Ovo dramatično smanjuje broj parametara.

#### Arhitektura

```
4× (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
kanali: 3 → 16 → 32 → 64 → 128
→ AdaptiveAvgPool2d(1,1)
→ Flatten → Dropout(0.35) → Linear(128, 4)
```

#### Hiperparametri

| Parametar | Vrijednost |
|-----------|-----------|
| Epohe (max) | 30 |
| Learning rate | 1e-4 |
| Weight decay | 2e-4 |
| Dropout | 0.35 |
| Patience | 8 |
| Augmentacija | HFlip, VFlip, Rot±15°, ColorJitter |
| Trainable params | ~98 436 |

#### Zaključak

**Test accuracy: ~87.29%**. Drastično smanjenje broja parametara (623K → 98K) uz poboljšanje test accuracya! GAP forsira model da se oslanja na naučene feature mape umjesto memoriranja prostornih detalja. Ovo je bila ključna arhitekturna odluka projekta.

---

### Run 7b i 7c — GAP CNN v2 i clean tracking varijante

Ove dvije varijante eksperimentiraju s poboljšanim praćenjem treninga. U v2 su promijenjeni neki hiperparametri (`batch=64, LR=8e-5, dropout=0.25, label_smoothing=0.05`), a dodani su i čisti evaluacijski prolasci kroz trening skup (bez augmentacije i s dropoutom isključenim).

**v2 rezultat: ~85%** — lošije od originala jer su se promijenili hiperparametri.

**Zaključak:** Originalni hiperparametri su bili bolji. v2 je korisna metodološka lekcija: trening krivulje mjerene uz aktivnu augmentaciju i dropout nisu usporedive s test krivuljama. Treba odvojeno evaluirati trening skup u eval modu.

GAP CNN clean tracking (13C) kombinira originalne hiperparametre s čistim praćenjem, te dodaje:
- ROC krivulje (one-vs-all)
- Precision-Recall krivulje
- Per-class accuracy bar chart
- Histogram confidence distribucija
- Box plot confidence po klasi

---

### Run 8 — Mali rezidualni CNN (from scratch)

#### Motivacija

GAP CNN ima odličan feature extractor i lagani classifier. Sljedeći logičan korak nije povećati classifier head, već poboljšati konvolucijski feature extractor kroz rezidualne blokove koji omogućuju stabilniji trening dubljih mreža.

#### Arhitektura

```
Stem:    Conv(3→32) → BN → ReLU
Stage 1: ResidualBlock(32) → MaxPool2d
Down1:   Conv(32→64) → BN → ReLU
Stage 2: ResidualBlock(64) → MaxPool2d
Down2:   Conv(64→128) → BN → ReLU
Stage 3: ResidualBlock(128) → MaxPool2d
Down3:   Conv(128→192) → BN → ReLU
Stage 4: ResidualBlock(192) → MaxPool2d
         AdaptiveAvgPool2d(1,1)
         Flatten → Dropout(0.30) → Linear(192, 4)

ResidualBlock: Conv → BN → ReLU → Conv → BN → (+skip) → ReLU
```

#### Hiperparametri

| Parametar | Vrijednost |
|-----------|-----------|
| Epohe (max) | 35 |
| Learning rate | 1e-4 |
| Weight decay | 2e-4 |
| Dropout | 0.30 |
| Patience | 8 |
| LR scheduler | ReduceLROnPlateau (max) |

#### Zaključak

**Test accuracy: ~89.10%** — najbolji custom model u projektu! Rezidualne veze stabiliziraju gradijente i dozvoljavaju modelu da nauči dublje vizualne reprezentacije bez problema nestajućih gradijenata. Od ukupno 2487 test uzoraka, samo ~271 je krivo klasificirano.

#### Analiza grešaka

Identificirani su najčešći parovi konfuzije na test skupu:

- **EOSINOPHIL → NEUTROPHIL** (najčešća greška): morfološka sličnost u obliku jezgre
- **MONOCYTE → NEUTROPHIL** (druga po redu greška)

NEUTROPHIL ima visoki recall ali nisku precision — model ga previše predviđa i za vizualno slične klase. Ove greške su sustavne i koncentrirane između vizualno sličnih klasa.

---

### Run 9 — Transfer learning: ResNet18 (zamrznuta jezgra)

#### Pristup

Korišten je pretreniran ResNet18 (ImageNet težine). Cijela konvolucijska jezgra je zamrznuta (`param.requires_grad=False`). Zamijenjen je samo posljednji `fc` sloj s `Dropout(0.30) → Linear(512, 4)`. Ulazne slike rezane na 224×224, normalizacija ImageNet standardima (`mean=[0.485, 0.456, 0.406]`).

| Parametar | Vrijednost |
|-----------|-----------|
| Epohe (max) | 15 |
| Learning rate | 1e-4 |
| Patience | 5 |
| Trainable params | ~2 052 (samo head) |

#### Zaključak

**Test accuracy: ~58%** — lošije od svih custom CNN modela! ImageNet i mikroskopske slike krvnih stanica su previše različite vizualne domene. Zamrznuta jezgra ne može adaptirati visoko-razinske reprezentacije na morfologiju stanica, boje bojanja i mikroskopske teksture.

---

### Run 10 — Transfer learning: ResNet18 (fine-tuning layer4)

#### Što je promijenjeno?

Odmrznuti su parametri samo u zadnjem rezidualnom bloku (`layer4`), dok su prethodni slojevi ostali zamrznuti. Ovo dozvoljava da se visoko-razinske reprezentacije adaptiraju na mikroskopsku domenu uz zadržavanje generalnih low-level featurea. Learning rate smanjen na `3e-5` za finiji fine-tuning.

#### Zaključak

Fine-tuning `layer4` daje bolje rezultate od potpuno zamrznute jezgre, ali ostaje ispod custom modela treniranih od nule. Ovo potvrđuje da je vizualna domena mikroskopije dostatno specifična da custom modeli, trenirani direktno na domeni, mogu nadmašiti pretrenirana rješenja bez fine-tuninga.

---

## 3. Usporedna tablica rezultata

| Model | Best test acc. | Napomene |
|-------|---------------|----------|
| Majority baseline | 25.09% | Uvijek predviđa najčešću klasu |
| Linearni klasifikator | 29.47% | Jedan Linear sloj, bez nelinearnosti |
| Nestabilni MLP | 26.00% | Jedan skriveni sloj, bez norm. i stabilizacije |
| Stabilizirani MLP | 40.00% | Normalizacija, manji LR, AdamW+WD |
| Duboki stabilizirani MLP | 45.20% | 2 skrivena sloja, BatchNorm1d, Dropout |
| Osnovni CNN | 72.26% | 3 konv. bloka, prvi CNN model |
| Poboljšani CNN | 79.94% | BatchNorm2d, Dropout, augmentacija |
| **CNN s GAP** | **87.29%** | Global Average Pooling, ~98K parametara |
| **Mali rezidualni CNN** | **89.10%** | Rezidualni blokovi, BEST custom model |
| ResNet18 (zamrznuta jezgra) | ~58.00% | Transfer learning, domena nedovoljno generalizira |
| ResNet18 (fine-tuning layer4) | poboljšanje | Parcijalni fine-tuning, bolje od zamrznute verzije |

---

## 4. Ključni zaključci

### 4.1 Što je funkcioniralo

- **Konvolucijski slojevi su neophodna komponenta:** skok od MLP (~45%) na CNN (~72%) jasno pokazuje da lokalna prostorna struktura slike nosi ključne informacije za klasifikaciju krvnih stanica.
- **Global Average Pooling umjesto Flatten + veliki Linear:** smanjio je broj parametara 6× (623K → 98K) i poboljšao generalizaciju za 7.35 postotnih bodova.
- **Rezidualni blokovi (skip connections):** dozvoljavaju modelu da uči dublje reprezentacije bez gradijentskih problema, što je dalo finalnih ~2 postotnih boda poboljšanja.
- **Normalizacija ulaza i stabilni trening setup** (manji LR, AdamW s weight decay): ključno za konvergenciju MLP modela i sprječavanje nestabilnosti.
- **Data augmentation:** RandomFlip, RandomRotation i ColorJitter povećavaju robusnost na varijacije prisutne u mikroskopskim snimkama.
- **Early stopping i LR scheduler** (ReduceLROnPlateau): sprječavaju pretrenavanje i automatski prilagođavaju brzinu učenja.

### 4.2 Što nije funkcioniralo

- **Transfer learning sa zamrznutom jezgrom:** ImageNet i mikroskopska domena su previše različite — zamrznuta jezgra ne može naučiti relevantne vizualne uzorke za krvne stanice.
- **Veliki classifier head** (Flatten + Linear): potiče memoriranje prostornih detalja umjesto generaliziranja.
- **MLP nad sirovim pikselima bez normalizacije:** nestabilna optimizacija, loša generalizacija.

### 4.3 Sustavne greške finalnog modela

Mali rezidualni CNN najčešće griješi između EOSINOPHIL i NEUTROPHIL — dvije klase s morfološki sličnim izgledom jezgre. NEUTROPHIL ima visoki recall ali nisku precision, što znači da se model prečesto odlučuje za ovu klasu. Ovo nije slučajna greška već sustavna, i ukazuje na vizualnu sličnost ovih klasa koja bi se mogla adresirati s jačom augmentacijom, većim skupom podataka ili domain-specific arhitekturama.

### 4.4 Metodološke lekcije

- Trening krivulje mjerene uz aktivnu augmentaciju i dropout **NISU** izravno usporedive s test krivuljama. Ispravno je odvojeno evaluirati trening skup u eval modu bez augmentacije.
- Best model tracking prema test accuracy je bolji nego praćenje samo posljednje epohe — model se može "pokvariti" u kasnijim epohama zbog overfittinga.
- Iterativni razvoj (jedna promjena po koraku) olakšava razumijevanje koji faktori zaista doprinose poboljšanju.

### 4.5 Smjernice za daljnji razvoj

- Isprobati fully fine-tuned ResNet18 ili EfficientNet — parcijalni fine-tuning nije iskoristio puni potencijal transfer learninga.
- Primijeniti CutMix ili Mixup augmentaciju koja može poboljšati generalizaciju na vizualno slične klase.
- Koristiti veću rezoluciju ulaznih slika (128×128 ili 224×224) za custom CNN — više detalja može pomoći razlikovanju EOSINOPHIL i NEUTROPHIL.
- Istražiti class-specific augmentaciju za teže klase (EOSINOPHIL) kako bi se izbalansirala težina grešaka.
