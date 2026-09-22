**ver. 0.01-alpha**  
Initial version of multilayer perceptron for recognition of graphics.
This version contains alghoritm for recognising a number written on sketchpad

# 🧠 Višeslojni Perceptron (MLP) — Prepoznavanje rukopisnih znakova

Interaktivna veb aplikacija za prepoznavanje rukom pisanih cifara pomoću **višeslojnog perceptrona (Multilayer Perceptron — MLP)**, implementiranog **od nule (from scratch)** u čistom Python-u i biblioteci **NumPy**, bez korišćenja naprednih biblioteka poput TensorFlow-a ili PyTorch-a.

Aplikacija poseduje moderan grafički veb interfejs izrađen u Flask-u sa HTML5 Canvas-om na kojem možete slobodno crtati cifre, dok neuronska mreža u realnom vremenu vrši predikciju uz prikaz top 5 kandidata i raspodele verovatnoća.

---

## 📋 Sadržaj

- [Ključne karakteristike](#-ključne-karakteristike)
- [Da li je potrebno ponovo trenirati model?](#-da-li-je-potrebno-ponovo-trenirati-model)
- [Korišćene tehnologije](#-korišćene-tehnologije)
- [Inicijalni koraci i pokretanje](#-inicijalni-koraci-i-pokretanje)
- [Kako sistem funkcioniše](#-kako-sistem-funkcioniše)
  - [1. Arhitektura MLP mreže](#1-arhitektura-mlp-mreže)
  - [2. Obrada i centriranje crteža (Preprocessing)](#2-obrada-i-centriranje-crteža-preprocessing)
  - [3. Veb interfejs i API](#3-veb-interfejs-i-api)
- [Struktura projekta](#-struktura-projekta)
- [Ponovno treniranje modela (opciono)](#-ponovno-treniranje-modela-opciono)

---

## ✨ Ključne karakteristike

- **Spreman za rad odmah nakon preuzimanja**: Repozitorijum već sadrži pretreniran i verifikovan model sa **98.65% tačnosti** na testnom skupu.
- **Implementacija od nule**: Propagacija unapred (*forward pass*), propagacija unazad (*backpropagation*), optimizator, regularizacija i funkcija gubitka napisani su ručno uz matričnu algebru u NumPy-ju.
- **Adam optimizator & He inicijalizacija**: Za stabilno i brzo konvergiranje tokom obuke.
- **Dropout regularizacija**: Sprečava prenaučenost (*overfitting*) na skrivenim slojevima.
- **Napredna predobrada slike**: Automatsko uokvirivanje (*bounding box*), centriranje, dodavanje padding-a i Lanczos skaliranje na dimenziju $28 \times 28$ imitiraju proces kroz koji prolazi originalni MNIST skup.
- **Interaktivni Dark-mode interfejs**: Mogućnost podešavanja debljine četkice, trenutnog brisanja, kao i "Auto" režim koji prepoznaje znak čim završite potez.

---

## ❓ Da li je potrebno ponovo trenirati model?

> [!NOTE]
> **Kratak odgovor: NE, NEMA POTREBE!**
> Dovoljno je samo da klonirate ili preuzmete repozitorijum i odmah pokrenete aplikaciju.

### Detaljnije objašnjenje:  
**Pretrenirani model je već uključen**: U folderu `models/` nalaze se fajlovi:
   - `models/mlp_model.pkl` — serijalizovani model sa naučenim težinama i pomacima (weights & biases).
   - `models/metadata.json` — metapodaci o strukturi mreže i postignutim rezultatima.


---

## 🛠️ Korišćene tehnologije

| Tehnologija | Uloga u projektu |
| :--- | :--- |
| **Python 3** | Primarni programski jezik |
| **NumPy** | Matrični proračuni, tenzorske operacije, He inicijalizacija, Adam optimizator |
| **Flask** | Backend veb server i REST API rute za inferenciju i kontrolu |
| **Pillow (PIL)** | Predobrada slike sa canvas-a (crop, centriranje, grayscale, blur, resize) |
| **HTML5 Canvas / CSS3 / Vanilla JS** | Grafički korisnički interfejs, crtanje poteza mišem/dodirom i asinhrona komunikacija (`fetch`) |
| **MNIST Dataset** | Skup od 70.000 slika rukom pisanih cifara (60.000 trening + 10.000 test) |

---

## 🚀 Inicijalni koraci i pokretanje

Sve što je potrebno jeste instalirati Python i zavisnosti, a potom pokrenuti server.

### 1. Kloniranje repozitorijuma
```bash
git clone https://github.com/vladankovich/MLP-graph.git
cd MLP-graph
```

### 2. (Opciono) Kreiranje virtuelnog okruženja
Preporučuje se rad u virtuelnom okruženju:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalacija biblioteka
Instalirajte potrebne pakete navedene u `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Zavisnosti su minimalne: `flask`, `numpy`, `pillow`)*

### 4. Pokretanje veb aplikacije
Pokrenite glavni server:
```bash
python app.py
```

U terminalu ćete videti poruku:
```text
============================================================
  MLP Demo - Prepoznavanje rukopisnih znakova
============================================================
  Ucitavanje modela...
  Model ucitan: 10 klasa, tacnost=98.65%
  Model uspjesno ucitan!

  Server pokrenut na: http://localhost:5000
============================================================
```

### 5. Otvaranje u browseru
Otvorite vaš omiljeni internet pregledač i idite na:
👉 **[http://localhost:5000](http://localhost:5000)** (ili `http://127.0.0.1:5000`)

---

## 🔬 Kako sistem funkcioniše

### 1. Arhitektura MLP mreže

Mreža se sastoji od 5 slojeva (ulazni, 3 skrivena i izlazni):

```
Ulaz (784 piksela)
       │
       ▼
[ Skriveni sloj 1: 512 neurona ] ──► ReLU ──► Dropout (p=0.3)
       │
       ▼
[ Skriveni sloj 2: 256 neurona ] ──► ReLU ──► Dropout (p=0.3)
       │
       ▼
[ Skriveni sloj 3: 128 neurona ] ──► ReLU ──► Dropout (p=0.2)
       │
       ▼
[ Izlazni sloj: 10 neurona ]      ──► Softmax ──► Verovatnoće [0..9]
```

- **Aktivacione funkcije**: 
  - **ReLU** ($ReLU(z) = \max(0, z)$) na skrivenim slojevima (sprečava iščezavanje gradijenta).
  - **Softmax** na izlazu (pretvara logite u pravu raspodelu verovatnoće čiji je zbir 1).
- **Optimizator (Adam)**: Koristi prvi ($m$) i drugi moment ($v$) gradijenata sa eksponencijalnim opadanjem ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) uz opadanje stope učenja (*learning rate decay* = 0.97 po epohi).
- **Gubitak**: Kategorička unakrsna entropija (*Categorical Cross-Entropy*).

### 2. Obrada i centriranje crteža (Preprocessing)

Model treniran na MNIST skupu je osetljiv na poziciju i debljinu linija. Iz tog razloga, pre slanja u mrežu, slika sa canvas-a prolazi kroz sledeći lanac u `preprocess_canvas_image()`:
1. **Prijem crteža**: Canvas izvozi sliku u Base64 PNG formatu.
2. **Alfa kompozit & Grayscale**: Potezi se crtaju na crnoj pozadini sa belim mastilom.
3. **Gaussian Blur (0.5)**: Blago omekšavanje oštrih ivica radi vernije simulacije rukopisa.
4. **Centriranje gravitacije / Bounding Box**: Iseca se samo nacrtana cifra, dodaje se ravnomerni padding (20%) i postavlja u centar kvadratnog platna.
5. **Lanczos Skaliranje**: Slika se smanjuje na dimenzije $28 \times 28$ piksela.
6. **Normalizacija**: Vrednosti piksela $[0, 255]$ se dele sa $255.0$ u opseg $[0.0, 1.0]$ i ravnaju u vektor dimenzije $(1, 784)$.

### 3. Veb interfejs i API

- `GET /` — Servira glavni HTML/JS korisnički interfejs.
- `GET /api/status` — Vraća podatke o tome da li je model spreman, njegovoj arhitekturi i tačnosti.
- `POST /api/predict` — Prima JSON sa Base64 slikom i vraća predviđenu klasu, procenat pouzdanosti i top 5 verovatnoća.
- `POST /api/train` — Pokreće asinhrono treniranje u pozadinskoj niti (`threading.Thread(daemon=True)`).
- `GET /api/training_progress` — Vraća podatke o trenutnoj epohi i gubicima za prikaz *progress bar*-a u realnom vremenu.

---

## 📁 Struktura projekta

```text
MLP_Demo/
├── app.py                     # Flask veb server i rute za predikciju/status
├── mlp.py                     # Implementacija MLP-a, He inicijalizacije, Adam-a i slojeva
├── train.py                   # Skript za treniranje modela i evaluaciju
├── data_loader.py             # Učitavanje i obrada MNIST i EMNIST skupova podataka
├── requirements.txt           # Potrebne Python biblioteke
├── README.md                  # Dokumentacija projekta
├── data/                      # Skupovi podataka
│   ├── mnist.npz              # MNIST baza cifara (60k trening + 10k test)
│   └── emnist-letters-*.gz    # EMNIST baza slova (opciono za proširenje)
├── models/                    # Sačuvani modeli i statistike
│   ├── mlp_model.pkl          # Trenirani model (težine i pomaci)
│   ├── metadata.json          # Podaci o tačnosti, slojevima i obuci
│   └── training_progress.json # Poslednji zabeleženi progres obuke
└── templates/
    └── index.html             # Frontend aplikacije (HTML, CSS i JS)
```

---

## 🔄 Ponovno treniranje modela (opciono)

Ukoliko ikada poželite da ponovite proces obuke od nule:

### Način A: Kroz terminal
```bash
python train.py
```
Ovo će pokrenuti obuku na 60 epoha, ispisivati gubitak i tačnost po epohama, i na kraju pregaziti fajlove u `models/` novim vrednostima.

### Način B: Kroz veb interfejs
1. Otvorite aplikaciju na `http://localhost:5000`.
2. U desnom donjem delu kliknite na dugme **"🚀 Pokreni treniranje"**.
3. U realnom vremenu možete pratiti napredak po epohama, kretanje funkcije gubitka i tačnosti.

---
