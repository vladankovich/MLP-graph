"""
Skript za obučavanje (treniranje) višeslojnog perceptrona.
Usklađeno sa završnim (diplomskim) radom:
"Višeslojni perceptron neuronskih mreža u prepoznavanju slika"
Kandidat: Vladan Kenjić | Mentor: Doc. dr Maid Omerović
Fakultet za tehničke studije (FTS), Univerzitet u Travniku

Arhitektura modela (Poglavlje V, P280-P288):
    - Ulaz: 784 neurona (28x28 normalizovano u [0, 1])
    - Prvi skriveni sloj: 128 neurona (ReLU) -> 100.480 parametara
    - Drugi skriveni sloj: 64 neurona (ReLU) -> 8.256 parametara
    - Izlazni sloj: 10 neurona (Softmax) -> 650 parametara
    - Ukupno obučivih parametara: 109.386

Hiperparametri obučavanja (Poglavlje V, P289):
    - Optimizator: Adam (beta1=0.9, beta2=0.999, epsilon=1e-8)
    - Početna stopa učenja (eta): 0.001 (sa blagim decay-om 0.97 po epohi)
    - Veličina mini-paketa (batch size): 64 uzorka
    - Broj epoha: 40
    - Funkcija gubitka: Kategorička unakrsna entropija (Categorical Cross-Entropy)
    - Regularizacija: Dropout (p=0.2 na skrivenim slojevima)
"""

import numpy as np
import os
import sys
import json
import time

# Dodavanje tekućeg direktorijuma u putanju modula
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp import MLP
from data_loader import load_dataset, one_hot

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train(epochs=40, batch_size=64, lr=0.001, lr_decay=0.97, seed=42):
    """
    Glavna funkcija obučavanja MLP modela usklađena sa parametrima rada.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 75)
    print("  POKRETANJE OBUČAVANJA VISESLOJNOG PERCEPTRONA (MLP)")
    print("  Tema: Klasifikacija rukopisnih cifara na MNIST skupu podataka")
    print("=" * 75)

    start_time = time.time()

    # 1. Učitavanje i normalizacija podataka
    X_train, y_train, X_test, y_test, n_classes, class_names = load_dataset()

    # One-hot kodovanje oznaka klasa
    y_train_oh = one_hot(y_train, n_classes)
    y_test_oh = one_hot(y_test, n_classes)

    print(f"\n  Podaci o skupu:")
    print(f"    Broj slika za obučavanje: {X_train.shape[0]:,}")
    print(f"    Broj slika za testiranje: {X_test.shape[0]:,}")
    print(f"    Broj klasa:               {n_classes} (cifre 0 do 9)")

    # 2. Definisanje projektovane arhitekture
    layer_sizes = [784, 128, 64, 10]
    dropout_rates = [0.2, 0.2, 0.0]

    model = MLP(layer_sizes=layer_sizes, dropout_rates=dropout_rates, seed=seed)
    total_params, param_details = model.count_parameters()

    print(f"\n  Projektovana arhitektura mreže (Poglavlje V rada):")
    layer_names = [
        "Ulazni sloj (Linearizacija 28x28)",
        "Prvi skriveni sloj (Dense + ReLU)",
        "Drugi skriveni sloj (Dense + ReLU)",
        "Izlazni sloj (Dense + Softmax)"
    ]
    for i, name in enumerate(layer_names):
        size = layer_sizes[i]
        if i == 0:
            print(f"    {name:38s}: {size} neurona (nema parametara)")
        else:
            p_info = param_details[i - 1]
            print(f"    {name:38s}: {size} neurona | {p_info['weights']:,} težina + {p_info['biases']} pomjeraja = {p_info['total']:,} parametara")

    print(f"\n    >> UKUPAN BROJ PARAMETARA MREŽE: {total_params:,} <<")
    assert total_params == 109386, f"Broj parametara {total_params} mora biti tačno 109.386!"

    print(f"\n  Hiperparametri obučavanja:")
    print(f"    Optimizator:            Adam (beta1=0.9, beta2=0.999, epsilon=1e-8)")
    print(f"    Funkcija gubitka:       Kategorička unakrsna entropija (Categorical Cross-Entropy)")
    print(f"    Veličina mini-paketa:   {batch_size} (mini-batch)")
    print(f"    Broj epoha:             {epochs}")
    print(f"    Početna stopa učenja:   {lr} (faktor opadanja {lr_decay} po epohi)")
    print(f"    Dropout regularizacija: {dropout_rates}")

    # Praćenje progresa za real-time JSON API
    train_history = []

    def epoch_callback(record):
        train_history.append(record)
        with open(os.path.join(MODELS_DIR, "training_progress.json"), "w") as f:
            json.dump({
                "history": train_history,
                "total_epochs": epochs,
                "current_epoch": record["epoch"],
                "status": "training"
            }, f)

    # 3. Pokretanje obučavanja
    print(f"\n  Počinje proces obučavanja kroz {epochs} epoha...\n")
    history = model.fit(
        X_train, y_train_oh,
        X_val=X_test, y_val=y_test_oh,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lr_decay=lr_decay,
        callback=epoch_callback
    )

    # 4. Finalna evaluacija nad testnim skupom (10.000 slika)
    print("\n" + "=" * 75)
    print("  FINALNA EVALUACIJA MODELA NAD NEVIĐENIM TESTNIM SKUPOM (10.000 SLIKA)")
    print("=" * 75)

    duration = time.time() - start_time
    train_acc = model.accuracy(X_train, y_train_oh)
    test_acc = model.accuracy(X_test, y_test_oh)
    error_rate = 1.0 - test_acc
    n_errors = int(round(error_rate * len(X_test)))

    print(f"    Tačnost na trening skupu (60.000): {train_acc * 100:.2f}%")
    print(f"    Tačnost na testnom skupu (10.000): {test_acc * 100:.2f}%")
    print(f"    Stopa greške na testu:             {error_rate * 100:.2f}%")
    print(f"    Ukupan broj pogrešnih slika:       {n_errors} od {len(X_test)}")
    print(f"    Ukupno trajanje obuke:             {duration:.1f} sekundi ({duration / 60:.1f} min)")

    # 5. Računanje matrice konfuzije (Tabela 1 u radu)
    print(f"\n  Generisanje matrice konfuzije...")
    cm = model.compute_confusion_matrix(X_test, y_test)

    # Prikaz matrice konfuzije u konzoli
    print("\n  Matrica konfuzije (Tabela 1 iz završnog rada):")
    header = "      " + " ".join([f"P{j:>4}" for j in range(10)])
    print(header)
    print("     " + "-" * 55)
    for i in range(10):
        row_str = f"  S{i} |" + " ".join([f"{cm[i, j]:>5}" for j in range(10)])
        print(row_str)

    # 6. Čuvanje modela i svih metapodataka
    model_path = os.path.join(MODELS_DIR, "mlp_model.pkl")
    model.save(model_path)

    cm_data = {
        "classes": class_names,
        "matrix": cm.tolist(),
        "total_samples": int(np.sum(cm)),
        "correct_samples": int(np.trace(cm)),
        "errors": int(np.sum(cm) - np.trace(cm)),
        "accuracy": float(test_acc)
    }
    with open(os.path.join(MODELS_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm_data, f, indent=2)

    meta = {
        "model_name": "MLP_Demo_1.0",
        "description": "Višeslojni Perceptron usklađen sa završnim radom",
        "author": "Vladan Kenjić",
        "layer_sizes": layer_sizes,
        "dropout_rates": dropout_rates,
        "total_parameters": total_params,
        "parameter_details": param_details,
        "n_classes": n_classes,
        "class_names": class_names,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "error_rate": float(error_rate),
        "test_errors_count": n_errors,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "learning_rate_decay": lr_decay,
        "optimizer": "Adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "training_duration_seconds": float(duration)
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(MODELS_DIR, "training_progress.json"), "w") as f:
        json.dump({
            "history": train_history,
            "total_epochs": epochs,
            "current_epoch": epochs,
            "status": "done",
            "final_accuracy": float(test_acc),
            "error_rate": float(error_rate)
        }, f)

    print(f"\n  Svi artefakti uspješno sačuvani u: {MODELS_DIR}")
    print("=" * 75 + "\n")

    return model, meta


if __name__ == "__main__":
    train()
