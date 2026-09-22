"""
Flask web server za interaktivni demo prepoznavanja rukopisnih cifara (MNIST).
Usklađeno sa završnim (diplomskim) radom:
"Višeslojni perceptron neuronskih mreža u prepoznavanju slika"
Kandidat: Vladan Kenjić | Mentor: Doc. dr Maid Omerović
Fakultet za tehničke studije (FTS), Univerzitet u Travniku
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import os
import json
import io
import sys
from PIL import Image, ImageFilter
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Globalni model i metapodaci
model = None
metadata = None
training_thread = None


def load_model():
    """Učitavanje serijalizovanog modela i pripadajućih metapodataka."""
    global model, metadata
    model_path = os.path.join(MODELS_DIR, "mlp_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")

    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            model = MLP.load(model_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            total_params, _ = model.count_parameters()
            print(f"  [OK] Model uspješno učitan!")
            print(f"       Arhitektura: {metadata['layer_sizes']}")
            print(f"       Broj obučivih parametara: {total_params:,}")
            print(f"       Test tačnost: {metadata['test_accuracy'] * 100:.2f}% (Stopa greške: {metadata['error_rate'] * 100:.2f}%)")
            return True
        except Exception as e:
            print(f"  [GREŠKA] Učitavanje modela nije uspjelo: {e}")
    return False


def center_image(img_array):
    """
    Centriranje crteža unutar slike na bazi centra mase i uokvirivanja (Bounding Box),
    što oponaša standardni pripremni postupak primijenjen na MNIST bazi.
    """
    if img_array.max() == 0:
        return img_array

    # Detekcija aktivnih piksela poteza
    rows = np.any(img_array > 0.1, axis=1)
    cols = np.any(img_array > 0.1, axis=0)
    if not rows.any() or not cols.any():
        return img_array

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Izrezivanje samo nacrtane cifre
    cropped = img_array[rmin:rmax + 1, cmin:cmax + 1]
    h, w = cropped.shape

    # Dodavanje proporcionalnog padding-a oko cifre (oko 20-25%)
    pad = max(h, w) // 4
    size = max(h, w) + 2 * pad
    canvas = np.zeros((size, size), dtype=np.float32)

    row_off = (size - h) // 2
    col_off = (size - w) // 2
    canvas[row_off:row_off + h, col_off:col_off + w] = cropped
    return canvas


def preprocess_canvas_image(image_data_b64):
    """
    Kompletan cjevovod predobrade slike nacrtane na Canvas-u:
        1. Dekodiranje Base64 PNG formata
        2. Kompozitiranje na crnoj pozadini (bela cifra na crnoj pozadini)
        3. Konverzija u sivu skalu (Grayscale L)
        4. Primjena blagog Gausovog zamućenja radi omekšavanja poteza
        5. Centriranje objekta (Bounding Box padding)
        6. Lanczos skaliranje na standardni MNIST format 28x28 piksela
        7. Min-Maks normalizacija piksela na interval [0.0, 1.0]
        8. Linearizacija (Ravnanje / Flattening) u vektor dužine 784 elementa
    """
    if "," in image_data_b64:
        image_data_b64 = image_data_b64.split(",")[1]
    img_bytes = base64.b64decode(image_data_b64)

    # Otvaranje slike
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Crna pozadina
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    composite = Image.alpha_composite(background, img)
    gray = composite.convert("L")

    # Blagi antialiasing / blur
    gray = gray.filter(ImageFilter.GaussianBlur(radius=0.5))

    arr = np.array(gray, dtype=np.float32) / 255.0

    # Centriranje
    arr = center_image(arr)

    # Skaliranje na 28x28 piksela
    pil_img = Image.fromarray((arr * 255).astype(np.uint8))
    pil_resized = pil_img.resize((28, 28), Image.Resampling.LANCZOS)

    # Min-maks normalizacija i linearizacija
    final_arr = np.array(pil_resized, dtype=np.float32) / 255.0
    return final_arr.reshape(1, 784)


@app.route("/")
def index():
    """Glavna stranica aplikacije."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Vraća trenutni status modela, arhitekturu i performanse."""
    progress_path = os.path.join(MODELS_DIR, "training_progress.json")

    if model is None:
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    prog = json.load(f)
                if prog.get("status") == "training":
                    history = prog.get("history", [])
                    last = history[-1] if history else {}
                    return jsonify({
                        "status": "training",
                        "message": "Obučavanje modela je u toku...",
                        "epoch": last.get("epoch", 0),
                        "total_epochs": prog.get("total_epochs", 40),
                        "train_acc": last.get("train_acc", 0),
                        "val_acc": last.get("val_acc", 0),
                        "loss": last.get("loss", 0)
                    })
            except Exception:
                pass
        return jsonify({
            "status": "no_model",
            "message": "Model nije pronađen. Potrebno je pokrenuti obučavanje."
        })

    total_params, param_details = model.count_parameters()
    return jsonify({
        "status": "ready",
        "classes": metadata["class_names"],
        "n_classes": metadata["n_classes"],
        "layer_sizes": metadata["layer_sizes"],
        "total_parameters": total_params,
        "parameter_details": param_details,
        "train_accuracy": metadata.get("train_accuracy", 0),
        "test_accuracy": metadata["test_accuracy"],
        "error_rate": metadata["error_rate"],
        "test_errors_count": metadata.get("test_errors_count", 0),
        "epochs": metadata.get("epochs", 40),
        "batch_size": metadata.get("batch_size", 64),
        "learning_rate": metadata.get("learning_rate", 0.001)
    })


@app.route("/api/confusion_matrix")
def api_confusion_matrix():
    """Vraća matricu konfuzije modela izračunatu nad 10.000 testnih slika."""
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.json")
    if os.path.exists(cm_path):
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Matrica konfuzije nije dostupna"}), 404


@app.route("/api/training_progress")
def training_progress():
    """Prati napredak obučavanja u realnom vremenu."""
    progress_path = os.path.join(MODELS_DIR, "training_progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({"status": "no_data", "history": []})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Klasifikacija cifre sa Canvas-a."""
    if model is None:
        return jsonify({"error": "Model nije učitan"}), 503

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "Nema podataka o slici"}), 400

    try:
        X = preprocess_canvas_image(data["image"])

        # Provjera da li je platno prazno
        if X.max() < 0.05:
            return jsonify({"error": "Platno je prazno! Nacrtajte cifru (0-9)."}), 400

        # Unaprijedno prostiranje za dobijanje vjerovatnoća (Softmax)
        probs = model.predict_proba(X)[0]
        predicted_idx = int(np.argmax(probs))

        # Top-5 najvjerovatnijih kandidata
        top_indices = np.argsort(probs)[::-1][:5]
        top5 = [
            {
                "class": metadata["class_names"][i],
                "probability": float(probs[i]) * 100
            }
            for i in top_indices
        ]

        # Vjerovatnoće za svih 10 klasa (0-9)
        all_probs = [
            {
                "class": metadata["class_names"][i],
                "probability": float(p) * 100
            }
            for i, p in enumerate(probs)
        ]

        return jsonify({
            "prediction": metadata["class_names"][predicted_idx],
            "confidence": float(probs[predicted_idx]) * 100,
            "top5": top5,
            "all_probabilities": all_probs,
            "n_classes": metadata["n_classes"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def start_training():
    """Pokretanje obuke u pozadinskoj niti."""
    global training_thread

    if training_thread and training_thread.is_alive():
        return jsonify({"message": "Obučavanje je već u toku"}), 409

    def run_training():
        global model, metadata
        try:
            from train import train as do_train
            m, meta = do_train(epochs=40, batch_size=64, lr=0.001)
            model = m
            metadata = meta
        except Exception as e:
            import traceback
            print(f"Greška pri obučavanju: {e}")
            traceback.print_exc()

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    return jsonify({"message": "Obučavanje je uspješno pokrenuto u pozadini!"})


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MLP Demo 1.0 — Prepoznavanje rukopisnih cifara (MNIST)")
    print("  Višeslojni perceptron: 784 -> 128 -> 64 -> 10 (109.386 parametara)")
    print("=" * 70)
    print("  Učitavanje modela...")
    if load_model():
        print("  Model spreman za rad!")
    else:
        print("  UPOZORENJE: Model nije pronađen.")
        print("  Pokrenite 'python train.py' da obučite model.")
    print("\n  Aplikacija je dostupna na: http://localhost:5000")
    print("=" * 70 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
