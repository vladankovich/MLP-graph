# 🧠 Višeslojni Perceptron (MLP) — Prepoznavanje rukopisnih cifara (MNIST)
### Praktično programsko rješenje uz završni (diplomski) rad
**Tema rada:** Višeslojni perceptron neuronskih mreža u prepoznavanju slika  
**Verzija softvera:** `MLP_Demo_1.0` (Septembar 2026.)

---

## 📌 Pregled projekta
Sistem implementira model **višeslojnog perceptrona (Multilayer Perceptron — MLP)** potpuno **od nule (*from scratch*)**, oslanjajući se isključivo na linearnu algebru i matrične proračune u biblioteci **NumPy**. U projektu nisu korišćeni gotovi paketi visokog nivoa (poput TensorFlow-a, PyTorch-a ili Keras-a), već je svaka formula iz teorijskog i matematičkog dijela rada direktno pretočena u izvorni kod.

---

## 🎯 Ključne matematičke i arhitekturne karakteristike

| Parametar / Osobina | Specifikacija u radu | Implementacija u kodu (`MLP_Demo_1.0`) |
| :--- | :--- | :--- |
| **Baza podataka** | MNIST (cifre 0–9, 70.000 slika) | 60.000 obučavanje, 10.000 testiranje (`data_loader.py`) |
| **Normalizacija piksela** | Min-maks na interval $[0.0, 1.0]$ | $X / 255.0$ (`float32`) |
| **Linearizacija (Flattening)** | $28 \times 28 \to 784$ elementa | `reshape(-1, 784)` |
| **Ulazni sloj** | 784 neurona (nema parametara) | Sloj 0: 784 dimenzije |
| **Prvi skriveni sloj** | 128 neurona (Dense + ReLU) | $784 \times 128 + 128 = \mathbf{100.480}$ parametara |
| **Drugi skriveni sloj** | 64 neurona (Dense + ReLU) | $128 \times 64 + 64 = \mathbf{8.256}$ parametara |
| **Izlazni sloj** | 10 neurona (Dense + Softmax) | $64 \times 10 + 10 = \mathbf{650}$ parametara |
| **UKUPAN BROJ PARAMETARA** | **109.386 obučivih parametara** | **Tačno 109.386 parametara** (`count_parameters()`) |
| **Inicijalizacija težina** | He (Kaiming) normalna raspodjela | $W \sim \mathcal{N}\left(0, \sqrt{2 / n_{in}}\right)$, $b = \mathbf{0}$ |
| **Funkcija gubitka** | Kategorička unakrsna entropija | $\mathcal{L}_{CE}(y, \hat{y}) = -\sum_{k=1}^K y_k \log \hat{y}_k$ |
| **Optimizator** | Adam (Adaptive Moment Estimation) | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}, \eta = 0.001$ |
| **Veličina mini-paketa** | Mini-batch od 64 uzorka | `batch_size = 64` |
| **Broj epoha obučavanja** | 40 epoha | `epochs = 40` |
| **Regularizacija** | Invertovani Dropout ($p=0.2$) | `dropout_rates = [0.2, 0.2, 0.0]` |
| **Postignuta tačnost na testu**| **98,20%** (180 grešaka od 10.000) | **98,20%** |
| **Matrica konfuzije** | Tabela 1 u radu (10×10 heatmap) | JSON perzistencija i interaktivni web prikaz |

---

## 📐 Matematička formulacija u kodu

### 1. Unaprijedno prostiranje signala (Forward Pass)
Za svaki sloj $l \in \{1, 2, 3\}$:
$$z^{[l]} = a^{[l-1]} W^{[l]} + b^{[l]}$$
$$a^{[l]} = \varphi^{[l]}(z^{[l]})$$
- Za skrivene slojeve ($l=1, 2$): $\varphi(z) = \text{ReLU}(z) = \max(0, z)$
- Za izlazni sloj ($l=3$): $\varphi(z)_k = \text{Softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{10} e^{z_j}}$

### 2. Propagacija signala greške unazad (Backpropagation)
- Signal greške na izlaznom sloju:
$$\delta^{[L]} = \frac{1}{M} \left(a^{[L]} - y\right)$$
- Gradijenti težina i pomjeraja:
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \left(a^{[l-1]}\right)^T \delta^{[l]}, \qquad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_{m=1}^M \delta_m^{[l]}$$
- Propagacija signala u prethodni sloj primjenom lančanog pravila diferenciranja:
$$\delta^{[l-1]} = \left(\delta^{[l]} \left(W^{[l]}\right)^T\right) \odot \text{mask}^{[l-1]} \odot \text{ReLU}'\left(z^{[l-1]}\right)$$

---

## 📁 Struktura direktorijuma `MLP_Demo_1.0`

```
MLP_Demo_1.0/
├── data/
│   └── mnist.npz                <- Baza MNIST podataka (60k trening, 10k test)
├── models/
│   ├── mlp_model.pkl            <- Obučeni model (težine, pomjeraji i hiperparametri)
│   ├── metadata.json            <- Metapodaci o modelu (109.386 parametara, 98.20% tačnost)
│   ├── confusion_matrix.json    <- Matrica konfuzije 10x10 nad 10.000 testnih slika
│   └── training_progress.json   <- Istorijat obuke kroz 40 epoha
├── templates/
│   └── index.html               <- Interaktivni tamni veb interfejs (Canvas, vizuelizator, matrica)
├── app.py                       <- Flask REST API veb server i predobrada korisničkog crteža
├── data_loader.py               <- Učitavanje, min-maks normalizacija i ravnanje MNIST baze
├── mlp.py                       <- Matematičko srce sistema (MLP, Adam, Dropout, Backprop)
├── train.py                     <- Skripta za obučavanje kroz 40 epoha sa batch veličinom 64
├── test_mlp.py                  <- Automatizovani testovi (12 unakrsnih provjera)
├── requirements.txt             <- Minimalne programske zavisnosti (flask, numpy, pillow)
├── IZVJESTAJ_O_IZMJENAMA.md     <- Detaljan akademski izvještaj o izmjenama i usklađivanju
└── README.md                    <- Ova dokumentacija
```

---

## 🚀 Uputstvo za pokretanje i korišćenje

### 1. Instalacija zavisnosti
Otvorite terminal ili PowerShell u folderu `MLP_Demo_1.0`:
```bash
pip install -r requirements.txt
```

### 2. Pokretanje veb aplikacije
Model je već obučen i spreman za rad. Pokrenite server komandom:
```bash
python app.py
```
U terminalu će se ispisati:
```text
======================================================================
  MLP Demo 1.0 — Prepoznavanje rukopisnih cifara (MNIST)
  Višeslojni perceptron: 784 -> 128 -> 64 -> 10 (109.386 parametara)
======================================================================
  [OK] Model uspješno učitan!
       Arhitektura: [784, 128, 64, 10]
       Broj obučivih parametara: 109.386
       Test tačnost: 98.20% (Stopa greške: 1.80%)

  Aplikacija je dostupna na: http://localhost:5000
======================================================================
```

### 3. Otvaranje u pretraživaču
Otvorite web pregledač i posjetite:  
👉 **[http://localhost:5000](http://localhost:5000)**

Na interfejsu možete:
- Mišem ili prstom nacrtati bilo koju cifru od **0 do 9**
- Podesiti debljinu poteza četkice
- Koristiti dugme **"🔍 Prepoznaj cifru"** ili uključiti **"⚡ Auto režim"**
- Posmatrati proračunatu raspodjelu vjerovatnoća (Softmax)
- Kliknuti na **"Prikaži detalje"** u sekciji Matrice konfuzije i analizirati dijagonalu pogodaka i tipične morfološke greške ($9 \to 4$, $5 \to 3$, $7 \to 2$).

---

## 🧪 Pokretanje automatizovanih testova

Za pokretanje integrisanog testnog paketa koji provjerava svaku liniju, formulu i dimenziju u kodu:
```bash
python test_mlp.py
```
Očekivani izlaz:
```text
............
----------------------------------------------------------------------
Ran 12 tests in ~1.8s

OK
```
Testovi automatski verifikuju:
1. Egzaktan proračun od **109.386 parametara**
2. ReLU aktivaciju i njen prvi izvod
3. Numerički stabilan Softmax
4. Kategoričku unakrsnu entropiju
5. Dimenzije tenzora u unaprijednom i unazadnom prolazu
6. Ažuriranje težina i pomjeraja kroz Adam algoritam
7. Proračun matrice konfuzije
8. Serijalizaciju i perzistenciju modela
9. One-hot kodovanje
10. Granice i normalizaciju MNIST baze $[0.0, 1.0]$
11. Bounding-box centriranje i skaliranje na Canvas-u
12. Flask REST API rute (`/api/status`, `/`, `/api/confusion_matrix`)

---

## 👨‍🎓 Akademska napomena
Ovaj projekat služi kao zvanični prateći softver za odbranu završnog rada na **Fakultetu za tehničke studije (FTS), Univerzitet u Travniku**.
