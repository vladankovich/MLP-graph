"""
Ucitavanje podataka - MNIST cifre + EMNIST slova (sa fallback na samo cifre).
"""

import urllib.request
import numpy as np
import os
import gzip
import struct
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url, dest, desc="Preuzimanje", timeout=15):
    if os.path.exists(dest):
        print(f"  Fajl vec postoji: {os.path.basename(dest)}")
        return True
    tmp = dest + ".tmp"
    try:
        print(f"  {desc}: {url}")

        def progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size / total_size * 100, 100)
                sys.stdout.write(f"\r    Progres: {pct:.1f}%  ")
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
                        sys.stdout.write(f"\r    Progres: {pct:.1f}%  ")
                        sys.stdout.flush()
        print()
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print(f"\n  Greska: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def read_idx_images(filepath):
    with gzip.open(filepath, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051
        n = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def read_idx_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049
        n = struct.unpack(">I", f.read(4))[0]
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
    ensure_dir()
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    dest = os.path.join(DATA_DIR, "mnist.npz")
    print("\n[MNIST] Ucitavanje skupa cifara (0-9)...")
    if not download_file(url, dest, "MNIST", timeout=60):
        return None
    data = np.load(dest)
    X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(np.int32)
    X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_test = data["y_test"].astype(np.int32)
    print(f"  MNIST ucitan: {X_train.shape[0]} trening, {X_test.shape[0]} test")
    return X_train, y_train, X_test, y_test


def load_emnist_letters():
    ensure_dir()
    files = {
        "train_images": "emnist-letters-train-images-idx3-ubyte.gz",
        "train_labels": "emnist-letters-train-labels-idx1-ubyte.gz",
        "test_images": "emnist-letters-test-images-idx3-ubyte.gz",
        "test_labels": "emnist-letters-test-labels-idx1-ubyte.gz",
    }
    base = "https://biometrics.nist.gov/cs_links/EMNIST/"
    print("\n[EMNIST] Pokusaj preuzimanja skupa slova (A-Z)...")
    for key, fname in files.items():
        dest = os.path.join(DATA_DIR, fname)
        if not download_file(base + fname, dest, f"EMNIST {key}", timeout=20):
            print("  EMNIST nije dostupan - koristice se samo cifre.")
            return None
    try:
        X_train = read_idx_images(os.path.join(DATA_DIR, files["train_images"]))
        y_train = read_idx_labels(os.path.join(DATA_DIR, files["train_labels"]))
        X_test = read_idx_images(os.path.join(DATA_DIR, files["test_images"]))
        y_test = read_idx_labels(os.path.join(DATA_DIR, files["test_labels"]))
        # EMNIST slike su transponirane u odnosu na MNIST
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
        y_train = (y_train - 1).astype(np.int32)
        y_test = (y_test - 1).astype(np.int32)
        X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
        X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
        print(f"  EMNIST slova ucitana: {X_train.shape[0]} trening, {X_test.shape[0]} test")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"  Greska pri ucitavanju EMNIST: {e}")
        return None


def one_hot(y, n_classes):
    n = len(y)
    oh = np.zeros((n, n_classes), dtype=np.float32)
    oh[np.arange(n), y] = 1.0
    return oh


def load_dataset():
    """
    Ucitava MNIST + EMNIST letters (ako su dostupni).
    Ako EMNIST nije dostupan, koristi samo MNIST (10 klasa).
    """
    mnist = load_mnist()
    if mnist is None:
        raise RuntimeError("Nije moguce ucitati MNIST!")

    X_train_d, y_train_d, X_test_d, y_test_d = mnist
    emnist = load_emnist_letters()

    if emnist is not None:
        X_train_l, y_train_l, X_test_l, y_test_l = emnist
        y_train_l = y_train_l + 10
        y_test_l = y_test_l + 10
        X_train = np.vstack([X_train_d, X_train_l])
        y_train = np.concatenate([y_train_d, y_train_l])
        X_test = np.vstack([X_test_d, X_test_l])
        y_test = np.concatenate([y_test_d, y_test_l])
        n_classes = 36
        class_names = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
        print(f"  Kombinovani skup: {X_train.shape[0]} uzoraka, {n_classes} klasa (0-9, A-Z)")
    else:
        X_train = X_train_d
        y_train = y_train_d
        X_test = X_test_d
        y_test = y_test_d
        n_classes = 10
        class_names = [str(i) for i in range(10)]
        print(f"  Samo cifre: {X_train.shape[0]} uzoraka, {n_classes} klasa (0-9)")

    return X_train, y_train, X_test, y_test, n_classes, class_names


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, nc, cn = load_dataset()
    print(f"Ucitano: train={X_tr.shape}, test={X_te.shape}, klase={cn}")
