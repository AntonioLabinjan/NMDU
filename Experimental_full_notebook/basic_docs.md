# Perceptryx Echo – sažetak evaluacijskog notebooka

Ovaj dokument sažima razvojni tok eval notebooka za Perceptryx Echo, audio/speaker recognition komponentu sustava Perceptryx. Notebook nije zamišljen kao jedan linearni finalni skript, nego kao eksperimentalni dnevnik: kroz više chunkova/cellova postupno se prelazi od jednostavne klasifikacije govornika prema ozbiljnijem embedding-based i open-set biometrijskom sustavu.

Glavna ideja razvoja bila je: prvo dokazati da se iz audio snimki može naučiti razlikovati govornike, zatim uvesti realniju evaluaciju, zatim poboljšavati arhitekturu i regularizaciju, a na kraju prijeći na metric learning + FAISS pristup koji je bliži stvarnom production scenariju.

---

## Kratki summary po chunkovima

### Chunk 1 – Linearni baseline nad mel-spektrogramima
Prvi chunk postavlja najjednostavniji mogući baseline. Audio se učitava, pretvara u mel-spektrogram, interpolira na fiksnu veličinu 128x128 i zatim spljošti u vektor koji ulazi u linearni klasifikator. Model se trenira kao closed-set klasifikator nad 40 osoba.

Rezultat pokazuje vrlo visoku točnost na treningu, čak do 100%, ali to nije dokaz dobre generalizacije. Ovaj chunk je važan jer daje početni dokaz da podaci nose informaciju o identitetu govornika, ali istovremeno otvara problem overfittinga i nedostatka ozbiljne validacije.

### Chunk 2 – Prvi CNN pokušaj
U drugom chunku uvodi se konvolucijska neuronska mreža. Ideja je da mel-spektrogram nije samo običan vektor, nego 2D reprezentacija audia u kojoj lokalni obrasci nose informaciju o glasu. CNN bi zato trebao bolje hvatati strukturu spektrograma nego linearni model.

Performanse su bile slabe. Model se zadržava na niskoj točnosti i ne uči stabilno. To je pokazalo da sama zamjena linearnog modela CNN-om nije dovoljna. Potrebni su bolji split podataka, bolja arhitektura, bolji trening i realnija evaluacija.

### Chunk 3 – Veći ResNet-ish model, ali bez pravog splita
Treći chunk uvodi ozbiljniju ResNet-ish arhitekturu s više konvolucijskih blokova. Dodani su bolji ispisi, mjerenje vremena, ETA i test inferencije. Model se trenira nad većim brojem uzoraka po osobi.

Ključna napomena u kodu je da split još nije napravljen. To znači da se model i dalje procjenjuje na način koji ne daje realnu sliku generalizacije. Iako trening accuracy raste do oko 95%, random inference test može dati krivu osobu s niskim confidenceom. Ovo je bio prijelomni trenutak: visoka trening točnost nije dovoljna ako nema odvojenog train/validation/test skupa.

### Chunk 4 – Uvođenje train/validation/test splita
Četvrti chunk uvodi `train_test_split`, `TensorDataset`, `DataLoader`, validacijsku petlju i finalnu evaluaciju. Ovo je prvi ozbiljniji pokušaj da se model mjeri korektno, a ne samo po trening točnosti.

Nakon uvođenja splita performanse postaju niže i nestabilnije, što je očekivano. Model više ne može samo memorirati trening uzorke, nego mora generalizirati na neviđene snimke. Ovaj chunk pokazuje realniju težinu problema speaker recognitiona.

### Chunk 5 – VoiceNetSlim, augmentacija i regularizacija
Peti chunk uvodi custom `Dataset` klasu, augmentaciju i regularizaciju preko weight decay parametra. Arhitektura je namjerno tanja, s ciljem da se smanji overfitting i vidi može li manji model bolje generalizirati.

Ovaj eksperiment ide u smjeru kontroliranja kompleksnosti modela. Nakon što je postalo jasno da veći model može memorirati podatke, pokušava se disciplinirati učenje: manje parametara, augmentacija i bolji data pipeline.

### Chunk 6 – VoiceNetMedium kao najbolji closed-set pokušaj
Šesti chunk je označen kao “best so far”. Uvodi se srednje velika arhitektura s kanalima 32 → 64 → 128 → 128, jačom reprezentacijom i boljom obradom spektrograma. Dataset koristi šum, time shift i SpecAugment elemente.

Ovdje se vidi balans: model više nije preslab kao prvi CNN, ali nije ni nepotrebno ogroman. Cilj je dobiti dovoljno kapaciteta za razlikovanje govornika, uz augmentacije koje pomažu da model ne ovisi previše o jednoj konkretnoj snimci.

### Chunk 7 – Eksperimentalno logiranje i reporti
Sedmi chunk uvodi automatske run foldere, spremanje konfiguracije, early stopping i generiranje izvještaja. Dodaju se grafovi, confusion matrix i classification report.

Ovo je pomak iz “samo treniraj model” faze u fazu ozbiljnog eksperimentiranja. Svaki run dobiva vlastiti folder i trag rezultata, što omogućuje usporedbu eksperimenata. To je bitno jer se kod neuronskih mreža puno toga svodi na iteraciju i mjerenje, a bez logiranja se brzo izgubi što je zapravo radilo.

### Chunk 8 – Čišći reporting pipeline
Osmi chunk dodatno rafinira logging i report generation. Uvodi se spremanje najboljeg modela, early stopping, sanity check spektrograma, normalizirana matrica konfuzije i strukturiraniji finalni report.

Smjer razmišljanja ovdje je reproducibilnost. Nije dovoljno da jedan trening “ispadne dobro”; treba moći kasnije otvoriti run folder i vidjeti konfiguraciju, krivulje, greške i ponašanje modela.

### Chunk 9 – Dublji VoiceNetDeep model
Deveti chunk pokušava s dubljom arhitekturom, većim batch sizeom, dužim treningom, malo manjim learning rateom i jačim weight decayem. Augmentacija je izbalansirana da ne uništi vokalne karakteristike.

Ovaj eksperiment ispituje hipotezu da modelu treba veći kapacitet. Istovremeno se pazi da augmentacija ne bude preagresivna, jer kod prepoznavanja glasa previše maskiranja ili šuma može ukloniti baš one detalje koji nose identitet govornika.

### Chunk 10 – Optimizirani dataset i class weights
Deseti chunk zadržava dublji model, ali optimizira dataset pipeline i uvodi class weights. Class weights služe za ublažavanje problema ako neke osobe imaju više ili manje uzoraka od drugih.

Ovdje fokus nije samo na arhitekturi, nego i na kvaliteti trening procesa. Ako dataset nije balansiran, model može preferirati klase koje češće vidi. Class weights pokušavaju taj bias smanjiti.

### Chunk 11 – ResNet + SE attention eksperiment
Jedanaesti chunk uvodi kompleksniji model s residual blokovima, SEBlock attention mehanizmom i OneCycleLR schedulerom. U kodu je označen kao “kinda shit”, što sugerira da kompleksnost nije donijela očekivani dobitak.

Ovo je važan negativan eksperiment. Pokazuje da veća i fancy arhitektura nije automatski bolja. Kod manjeg ili specifičnog dataseta kompleksniji model može biti teži za treniranje, sporiji i skloniji overfittingu.

### Chunk 12 – NAS/Optuna eksperiment
Dvanaesti chunk pokušava Neural Architecture Search pomoću Optune. Dinamički se isprobavaju različite konfiguracije modela i hiperparametara kroz više trialova.

Ideja je bila automatizirati potragu za boljom arhitekturom. Međutim, komentar u kodu sugerira da Optuna u ovom slučaju nije bila praktično korisna. Ovo je realan zaključak: NAS može pomoći, ali često troši puno vremena, a ne mora dati dovoljno bolji rezultat od ručno vođenih eksperimenata.

### Chunk 13 – Još jedan ResNet + Attention pokušaj
Trinaesti chunk ponovno pokušava s ResNet + attention pristupom, ali strukturiranije i s kontroliranijim brojem epoha. Model ima više faza i residual blokove.

Ovo izgleda kao pokušaj da se prethodna kompleksna ideja ipak stabilizira. Logički, ovo je zadnji dio closed-set classifier faze prije prelaska na embedding-based pristup.

### Chunk 14 – Prvi triplet learning / embedding model
Četrnaesti chunk uvodi veliki konceptualni zaokret: umjesto da model direktno klasificira osobu, trenira se embedding model pomoću Triplet Loss pristupa. Model uči da snimke iste osobe budu blizu u embedding prostoru, a snimke različitih osoba daleko.

Ovo je puno bliže stvarnom voice recognition sustavu. Closed-set klasifikator uvijek mora izabrati jednu od poznatih osoba, dok embedding model omogućuje usporedbu, thresholding, dodavanje novih osoba u bazu i open-set logiku. U ovom chunku se koristi pretrained model iz ranijeg runa kao baza, što je logičan pokušaj transfera naučenih audio featurea.

### Chunk 15 – Nastavak triplet learning pristupa
Petnaesti chunk je varijanta istog triplet learning pipelinea, s definiranim early stopping pragovima, embedding dimenzijom 128 i Triplet Loss marginom. Cilj je dobiti stabilan embedding prostor koji se kasnije može koristiti za usporedbu glasova i FAISS pretragu.

Ovaj dio predstavlja prelazak iz “model kao klasifikator” u “model kao feature extractor”. To je bitno za Perceptryx Echo jer production sustav ne smije biti ograničen na statični broj klasa iz treninga.

### Chunk 16 – Direktna usporedba dva glasa
Šesnaesti chunk učitava trenirani embedding model i definira funkciju za usporedbu dvije audio datoteke. Računa se euklidska distanca i kosinusna sličnost između embeddinga.

Ovo je prvi praktični verification test: umjesto da se pita “koja je ovo klasa?”, sustav pita “jesu li ova dva glasa dovoljno slična?”. To je bliže biometrijskom načinu razmišljanja.

### Chunk 17 – Izgradnja FAISS indeksa za voice database
Sedamnaesti chunk gradi FAISS indeks nad embeddingima svih poznatih audio uzoraka. Uz indeks se sprema i metadata, odnosno informacija koji embedding pripada kojoj osobi i kojem fileu.

Ovo je production-orijentirani korak. FAISS omogućuje brzo pretraživanje velikog broja embeddinga, što je nužno ako Perceptryx Echo treba raditi kao skalabilni recognition sustav.

### Chunk 18 – Test usporedbe različitih osoba
Osamnaesti chunk pokreće usporedbu dvije konkretne audio snimke. Dobivena je velika euklidska distanca i vrlo niska sličnost, pa sustav zaključuje da se radi o različitim osobama.

Ovaj chunk validira osnovnu intuiciju embedding prostora: različiti govornici trebaju biti udaljeni. To je mali, ali važan sanity check prije veće evaluacije.

### Chunk 19 – FAISS search test
Devetnaesti chunk gradi indeks za 40 osoba i 4874 zapisa te pokreće pretragu za testnu snimku. Najbliži rezultati većinom pripadaju istoj osobi, s vrlo malim distancama.

Ovo pokazuje da embedding + FAISS pristup radi kao retrieval sustav. Umjesto jedne direktne klasifikacije, sustav dobiva listu najbližih susjeda i može na temelju njih donositi odluku.

### Chunk 20 – Full evaluation: open-set K1/K2 consensus
Dvadeseti chunk je najzreliji evaluacijski dio notebooka. Uvodi se open-set K1/K2 consensus logika: FAISS vraća više najbližih susjeda, a sustav prihvaća identitet samo ako dovoljno susjeda glasa za istu osobu i ako je prosječna distanca dovoljno niska.

Ovdje se prvi put ozbiljno analiziraju biometrijske metrike: ROC AUC, EER, FAR, FRR, F1, threshold sweep, DET curve, score distributions, calibration, latency i K1/K2 grid search. Finalni zabilježeni rezultati uključuju ROC AUC oko 0.865, EER oko 21.61%, legacy accuracy oko 0.901 i mean inference latency oko 15.92 ms. Consensus konfiguracija daje visoku accuracy i F1, ali ima veći FAR od legacy threshold pristupa, što je važan sigurnosni signal.

Najvažniji zaključak ovog chunka je da accuracy sama po sebi nije dovoljna. Za voice recognition treba gledati trade-off između FAR-a i FRR-a. Ako sustav želi biti sigurniji, može spustiti FAR, ali tada raste FRR, odnosno više stvarnih korisnika bude odbijeno.

---

## Logički smjer razvoja i iteracija

### 1. Prvo je trebalo dokazati da problem uopće ima signal
Početni linearni model i prvi CNN nisu bili zamišljeni kao finalno rješenje. Njihova svrha bila je provjeriti može li se iz audio snimki uopće izvući dovoljno informacija za razlikovanje govornika. Linearni model je pokazao da spektrogrami nose jak signal, ali je istovremeno otkrio opasnost memoriranja.

### 2. Trening accuracy se pokazao kao preoptimistična metrika
Rani modeli su mogli postići visoku točnost na trening podacima, ali to nije značilo da model stvarno prepoznaje glasove u generalnom smislu. Zbog toga je uveden train/validation/test split. Nakon toga performanse su pale, ali su postale realnije.

To je važan dio razvoja: lošiji, ali pošten rezultat vrijedi više od lijepog, ali varljivog trening accuracyja.

### 3. Slijedila je borba između kapaciteta i generalizacije
Nakon uvođenja splita isprobane su različite arhitekture: slim, medium, deep, ResNet-ish, attention i NAS pristupi. Ideja je bila pronaći balans između modela koji ima dovoljno kapaciteta da razlikuje govornike i modela koji se ne raspadne na novim snimkama.

Eksperimenti su pokazali da kompleksniji model nije nužno bolji. Neki pokušaji, posebno attention/SE i NAS varijante, nisu bili dovoljno korisni u odnosu na kompleksnost koju uvode.

### 4. Augmentacija i regularizacija postaju ključne
Kod prepoznavanja glasa nije dovoljno da model čuje jednu čistu snimku osobe. U stvarnosti će glas varirati po rečenici, tonu, mikrofonu, šumu, udaljenosti i okolini. Zato su dodani noise, time shift, SpecAugment, weight decay i class weights.

Smjer razmišljanja je bio: ako model vidi više varijacija tijekom treninga, manje će se vezati za jednu konkretnu snimku i bolje će generalizirati.

### 5. Closed-set klasifikacija nije dovoljna za pravi recognition sustav
Closed-set classifier uvijek mora izabrati jednu od poznatih osoba. To nije dobro za realan sustav jer u stvarnosti može doći nepoznata osoba. Sustav tada ne smije “nasilno” dodijeliti najbližu poznatu klasu.

Zbog toga se prelazi na embedding model i Triplet Loss. Model više ne uči samo oznake klasa, nego uči prostor sličnosti: isti govornici trebaju biti blizu, različiti govornici daleko.

### 6. FAISS uvodi skalabilnu pretragu
Nakon što model proizvodi embeddinge, FAISS omogućuje brzo traženje najbližih poznatih glasova. To je bitno za Perceptryx Echo jer sustav treba moći raditi s bazom korisnika i brzo uspoređivati nove snimke s postojećim zapisima.

Ovo je isti mentalni model kao kod face recognition dijela Perceptryxa: model služi kao ekstraktor embeddinga, a FAISS služi kao brza memorija/pretraga identiteta.

### 7. Finalna evaluacija prelazi s “točnosti” na biometrijske metrike
U zadnjem chunku notebook se ponaša više kao evaluacija biometrijskog sustava nego kao obični classification zadatak. Umjesto samo accuracyja gledaju se FAR, FRR, EER, ROC AUC, threshold stability, latency i ponašanje na različitim thresholdima.

Ovo je najvažniji konceptualni pomak. Za sigurnosni/identity sustav nije dovoljno reći “accuracy je 91%”. Treba znati koliko često sustav prihvati pogrešnu osobu, koliko često odbije pravu osobu i koji threshold daje prihvatljiv kompromis.

---

## Glavni zaključci

1. Mel-spektrogrami sadrže dovoljno informacija za razlikovanje govornika, ali jednostavni modeli lako memoriraju podatke.

2. Realna evaluacija zahtijeva odvojeni train/validation/test split. Bez toga se rezultati mogu činiti puno boljima nego što jesu.

3. Veća arhitektura nije automatski bolja. Najbolji smjer nije bio samo dodavati slojeve, nego pronaći dobar balans modela, augmentacije i evaluacije.

4. Za production voice recognition bolji je embedding-based pristup nego obična closed-set klasifikacija.

5. FAISS je logičan dodatak jer omogućuje brzo pretraživanje embedding baze i lakše skaliranje na više korisnika.

6. Open-set evaluacija je nužna jer sustav mora znati kada osoba nije u bazi.

7. Finalni rezultati su solidni za eksperimentalni studentski/prototipni sustav, ali još nisu “production-grade biometric security” razina. Posebno treba paziti na FAR, jer visoka accuracy može sakriti činjenicu da sustav ponekad prihvaća pogrešne osobe.

8. Latency rezultati su vrlo dobri za real-time upotrebu. Mean inference latency od oko 15.92 ms sugerira da je model dovoljno brz za praktične scenarije, dok je glavna tema za poboljšanje preciznost i sigurnosni thresholding.

---

## Kratka verzija za ubaciti u seminar

U ovom notebooku razvijen je i evaluiran Perceptryx Echo, audio komponenta sustava za prepoznavanje identiteta. Razvoj je krenuo od jednostavnih closed-set klasifikatora nad mel-spektrogramima, uključujući linearni model i CNN arhitekture. Rani eksperimenti pokazali su da audio podaci nose signal o identitetu govornika, ali i da se modeli lako pretreniraju ako se koristi samo trening točnost.

Zbog toga je uvedena realnija evaluacija s train/validation/test splitom, augmentacijom, regularizacijom, class weightovima i eksperimentalnim loggingom. Isprobane su različite arhitekture, uključujući slim, medium, deep i ResNet/attention varijante. Pokazalo se da veća kompleksnost nije automatski bolja te da je važniji balans između kapaciteta modela i generalizacije.

Nakon closed-set faze napravljen je prijelaz na embedding-based pristup pomoću Triplet Loss funkcije. Umjesto da model direktno klasificira osobu, on uči embedding prostor u kojem su snimke iste osobe blizu, a snimke različitih osoba udaljene. Takav pristup je prikladniji za stvarni recognition sustav jer omogućuje usporedbu glasova, dodavanje novih osoba i open-set detekciju nepoznatih korisnika.

Za brzo pretraživanje embeddinga izgrađen je FAISS indeks, čime sustav prelazi iz obične klasifikacije u skalabilni retrieval pipeline. Finalna evaluacija uključuje ROC AUC, EER, FAR, FRR, F1, threshold sweep, DET krivulju, distribucije scoreova, kalibraciju, latency mjerenja i K1/K2 consensus logiku. Dobiveni rezultati pokazuju da sustav ima solidne performanse za prototip, s ROC AUC oko 0.865 i vrlo dobrom latencijom, ali da FAR/FRR trade-off ostaje ključan izazov za sigurnu production primjenu.

