"""
Skript za treniranje MLP modela.
Preuzima podatke, trenira mrezu i cuva model.
"""

import numpy as np
import os
import sys
import json
import time

# Dodaj trenutni direktorij u path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp import MLP
from data_loader import load_dataset, one_hot

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train(epochs=60, batch_size=128, lr=0.001, lr_decay=0.97):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("  TRENIRANJE VISESLOJNOG PERCEPTRONA")
    print("  Prepoznavanje rukopisnih cifara i slova")
    print("=" * 70)

    # Ucitavanje podataka
    start_time = time.time()
    X_train, y_train, X_test, y_test, n_classes, class_names = load_dataset()

    # One-hot encoding
    y_train_oh = one_hot(y_train, n_classes)
    y_test_oh = one_hot(y_test, n_classes)

    print(f"\n  Info o skupu podataka:")
    print(f"    Trening uzorci: {X_train.shape[0]:,}")
    print(f"    Test uzorci: {X_test.shape[0]:,}")
    print(f"    Broj klasa: {n_classes}")
    print(f"    Klase: {class_names}")

    # Arhitektura mreze
    layer_sizes = [784, 512, 256, 128, n_classes]
    dropout_rates = [0.3, 0.3, 0.2, 0.0]

    print(f"\n  Arhitektura MLP mreze:")
    layer_names = ["Ulazni sloj", "Skriveni sloj 1 (ReLU)", "Skriveni sloj 2 (ReLU)", "Skriveni sloj 3 (ReLU)", "Izlazni sloj (Softmax)"]
    for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        arrow = " -> " if i < len(layer_sizes) - 1 else ""
        print(f"    {name}: {size} neurona{arrow}")

    print(f"\n  Hiperparametri:")
    print(f"    Optimizator: Adam (beta1=0.9, beta2=0.999)")
    print(f"    Inicijalizacija: He (kaiming)")
    print(f"    Stopa ucenja: {lr} (decay={lr_decay}/epoha)")
    print(f"    Batch velicina: {batch_size}")
    print(f"    Epohe: {epochs}")
    print(f"    Dropout: {dropout_rates}")

    # Kreiranje modela
    model = MLP(layer_sizes, dropout_rates=dropout_rates)

    print(f"\n  Pocinje treniranje...\n")

    # Historija treniranja za pracenje napretka
    train_history = []

    def epoch_callback(record):
        train_history.append(record)
        # Pisi progres u JSON fajl za pracenje u realnom vremenu
        with open(os.path.join(MODELS_DIR, "training_progress.json"), "w") as f:
            json.dump({
                "history": train_history,
                "total_epochs": epochs,
                "status": "training"
            }, f)

    # Treniranje
    history = model.fit(
        X_train, y_train_oh,
        X_val=X_test, y_val=y_test_oh,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_decay=lr_decay,
        callback=epoch_callback,
    )

    # Finalna evaluacija
    print("\n" + "=" * 70)
    train_acc = model.accuracy(X_train, y_train_oh)
    test_acc = model.accuracy(X_test, y_test_oh)
    error_rate = 1.0 - test_acc
    duration = time.time() - start_time

    print(f"  REZULTATI TRENIRANJA:")
    print(f"    Tačnost na trening skupu: {train_acc*100:.2f}%")
    print(f"    Tačnost na test skupu:    {test_acc*100:.2f}%")
    print(f"    Stopa greske:             {error_rate*100:.2f}%")
    print(f"    Trajanje treniranja:      {duration:.1f}s")
    print()

    if error_rate < 0.10:
        print(f"  CILJ POSTIGNUT: Stopa greske {error_rate*100:.2f}% < 10%!")
    else:
        print(f"  Upozorenje: Stopa greske {error_rate*100:.2f}% >= 10%")
        print(f"  Preporuka: Povecaj broj epoha ili prilagodi arhitekturu")

    # Cuvanje modela
    model_path = os.path.join(MODELS_DIR, "mlp_model.pkl")
    model.save(model_path)

    # Cuvanje metapodataka
    meta = {
        "layer_sizes": layer_sizes,
        "dropout_rates": dropout_rates,
        "n_classes": n_classes,
        "class_names": class_names,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "error_rate": float(error_rate),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_duration_seconds": float(duration),
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Finalni status
    with open(os.path.join(MODELS_DIR, "training_progress.json"), "w") as f:
        json.dump({
            "history": train_history,
            "total_epochs": epochs,
            "status": "done",
            "final_accuracy": float(test_acc),
            "error_rate": float(error_rate),
        }, f)

    print(f"\n  Metapodaci sacuvani: {meta_path}")
    print("=" * 70)

    return model, meta


if __name__ == "__main__":
    train()
