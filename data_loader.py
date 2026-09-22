"""
Učitavanje i predobrada podataka — MNIST baza rukopisnih cifara.
Usklađeno sa završnim (diplomskim) radom:
"Višeslojni perceptron neuronskih mreža u prepoznavanju slika"
Kandidat: Vladan Kenjić | Mentor: Doc. dr Maid Omerović
Fakultet za tehničke studije (FTS), Univerzitet u Travniku

Baza podataka MNIST (LeCun et al., 1998):
    - Skup za obučavanje: 60.000 slika formata 28x28 piksela
    - Skup za testiranje: 10.000 slika formata 28x28 piksela
    - Klase: 10 klasa (cifre od 0 do 9)
    - Normalizacija: Min-Maks svođenje na interval [0.0, 1.0]
    - Linearizacija (Ravnanje / Flattening): 28x28 -> 784 elementa
"""

import urllib.request
import numpy as np
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def ensure_dir():
    """Kreira folder data/ ako ne postoji."""
    os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url, dest, desc="Preuzimanje", timeout=30):
    """Pouzdan download fajla uz prikaz progresa."""
    if os.path.exists(dest):
        return True
    tmp = dest + ".tmp"
    try:
        print(f"  {desc}: {url}")

        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size * 100, 100)
                sys.stdout.write(f"\r    Progres preuzimanja: {pct:.1f}%  ")
                sys.stdout.flush()

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        req = urllib.request.Request(url)
        with opener.open(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(downloaded / total * 100, 100)
                        sys.stdout.write(f"\r    Progres preuzimanja: {pct:.1f}%  ")
                        sys.stdout.flush()
        print()
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print(f"\n  Greška pri preuzimanju: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def one_hot(y, n_classes=10):
    """
    Konverzija cjelobrojnih oznaka klasa u one-hot kodovane vektore.
    Na primjer: cifra 7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    """
    n = len(y)
    oh = np.zeros((n, n_classes), dtype=np.float64)
    oh[np.arange(n), y] = 1.0
    return oh


def load_mnist():
    """
    Učitava standardni MNIST skup podataka.
    Vraća normalizovane i linearizovane podatke:
        X_train: (60000, 784) float32 u rasponu [0.0, 1.0]
        y_train: (60000,) int32 (klase 0-9)
        X_test:  (10000, 784) float32 u rasponu [0.0, 1.0]
        y_test:  (10000,) int32 (klase 0-9)
    """
    ensure_dir()
    dest = os.path.join(DATA_DIR, "mnist.npz")
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

    if not os.path.exists(dest):
        print("\n[MNIST] Preuzimanje MNIST skupa sa repozitorijuma...")
        success = download_file(url, dest, "MNIST baza (npz format)", timeout=60)
        if not success:
            raise RuntimeError(f"Nije uspjelo preuzimanje MNIST sa adrese: {url}")

    print("  Učitavanje MNIST baze...")
    data = np.load(dest)

    # Linearizacija (Flattening) 28x28 -> 784 i min-maks normalizacija na [0.0, 1.0]
    X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(np.int32)

    X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test = data["y_test"].astype(np.int32)

    print(f"  MNIST uspješno učitan: {X_train.shape[0]:,} trening slika, {X_test.shape[0]:,} test slika.")
    return X_train, y_train, X_test, y_test


def load_dataset():
    """
    Učitavanje kompletnog skupa podataka za potrebe obučavanja i evaluacije.
    Usklađeno sa Poglavljem V i VI završnog rada (10 klasa cifara 0-9).
    """
    X_train, y_train, X_test, y_test = load_mnist()
    n_classes = 10
    class_names = [str(i) for i in range(10)]
    return X_train, y_train, X_test, y_test, n_classes, class_names


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, nc, cn = load_dataset()
    print(f"  Oblik trening podataka: {X_tr.shape}")
    print(f"  Oblik test podataka:    {X_te.shape}")
    print(f"  Raspon vrijednosti piksela: [{X_tr.min():.2f}, {X_tr.max():.2f}]")
    print(f"  Definisane klase ({nc}): {cn}")
