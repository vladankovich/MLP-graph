"""
Flask web server za demo prepoznavanja znakova pomocu MLP mreze.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import base64
import os
import json
import io
import sys
from PIL import Image, ImageFilter, ImageOps
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
    global model, metadata
    model_path = os.path.join(MODELS_DIR, "mlp_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            model = MLP.load(model_path)
            with open(meta_path) as f:
                metadata = json.load(f)
            print(f"  Model ucitan: {metadata['n_classes']} klasa, tacnost={metadata['test_accuracy']*100:.2f}%")
            return True
        except Exception as e:
            print(f"  Greska pri ucitavanju modela: {e}")
    return False


def center_image(img_array):
    """Centriranje slike kao sto to radi MNIST preprocessing."""
    if img_array.max() == 0:
        return img_array
    # Pronalazak bounding boxa
    rows = np.any(img_array > 0.1, axis=1)
    cols = np.any(img_array > 0.1, axis=0)
    if not rows.any() or not cols.any():
        return img_array
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # Izrezivanje
    cropped = img_array[rmin:rmax+1, cmin:cmax+1]
    h, w = cropped.shape
    # Dodavanje paddinga (20% na svakoj strani)
    pad = max(h, w) // 4
    size = max(h, w) + 2 * pad
    canvas = np.zeros((size, size), dtype=np.float32)
    row_off = (size - h) // 2
    col_off = (size - w) // 2
    canvas[row_off:row_off+h, col_off:col_off+w] = cropped
    return canvas


def preprocess_canvas_image(image_data_b64):
    """
    Predobrada slike sa canvas-a za ulaz u MLP.

    Koraci:
    1. Dekodiranje base64 PNG slike
    2. Konverzija u sivu skalu (grayscale)
    3. Centriranje crteža unutar slike
    4. Skaliranje na 28x28 piksela
    5. Normalizacija piksela u [0, 1]
    """
    # Dekodiranje base64
    if "," in image_data_b64:
        image_data_b64 = image_data_b64.split(",")[1]
    img_bytes = base64.b64decode(image_data_b64)

    # Otvaranje slike
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Kompozitiranje na crnoj pozadini
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    composite = Image.alpha_composite(background, img)
    gray = composite.convert("L")

    # Primjena malog Gaussian blura za gladi potez
    gray = gray.filter(ImageFilter.GaussianBlur(radius=0.5))

    arr = np.array(gray, dtype=np.float32) / 255.0

    # Centriranje
    arr = center_image(arr)

    # Konverzija nazad u PIL sliku za resize
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = pil.resize((28, 28), Image.LANCZOS)

    # Finalna normalizacija
    final = np.array(pil, dtype=np.float32) / 255.0
    return final.reshape(1, 784)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Status modela i informacije o njemu."""
    # Provjeri da li je treniranje u toku
    progress_path = os.path.join(MODELS_DIR, "training_progress.json")

    if model is None:
        # Provjeri da li je treniranje u toku
        if os.path.exists(progress_path):
            try:
                with open(progress_path) as f:
                    prog = json.load(f)
                if prog.get("status") == "training":
                    history = prog.get("history", [])
                    last = history[-1] if history else {}
                    return jsonify({
                        "status": "training",
                        "message": "Model se trenira...",
                        "epoch": last.get("epoch", 0),
                        "total_epochs": prog.get("total_epochs", 60),
                        "train_acc": last.get("train_acc", 0),
                        "val_acc": last.get("val_acc", 0),
                        "loss": last.get("loss", 0),
                    })
            except Exception:
                pass
        return jsonify({
            "status": "no_model",
            "message": "Model nije pronaden. Pokreni train.py."
        })

    return jsonify({
        "status": "ready",
        "classes": metadata["class_names"],
        "n_classes": metadata["n_classes"],
        "layer_sizes": metadata["layer_sizes"],
        "train_accuracy": metadata.get("train_accuracy", 0),
        "test_accuracy": metadata["test_accuracy"],
        "error_rate": metadata["error_rate"],
    })


@app.route("/api/training_progress")
def training_progress():
    """Pracenje napretka treniranja u realnom vremenu."""
    progress_path = os.path.join(MODELS_DIR, "training_progress.json")
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({"status": "no_data", "history": []})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predikcija klase iz slike sa canvas-a."""
    if model is None:
        return jsonify({"error": "Model nije ucitan"}), 503

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "Nema podataka o slici"}), 400

    try:
        X = preprocess_canvas_image(data["image"])

        # Provjeri da li je canvas prazan
        if X.max() < 0.05:
            return jsonify({"error": "Canvas je prazan! Nacrtaj broj ili slovo."}), 400

        probs = model.predict_proba(X)[0]
        predicted_idx = int(np.argmax(probs))

        # Top-5 predikcije
        top_indices = np.argsort(probs)[::-1][:5]
        top5 = [
            {
                "class": metadata["class_names"][i],
                "probability": float(probs[i]) * 100
            }
            for i in top_indices
        ]

        # Sve klase s vjerovatnocama
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
            "n_classes": metadata["n_classes"],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def start_training():
    """Pokretanje treniranja u pozadini."""
    global training_thread

    if training_thread and training_thread.is_alive():
        return jsonify({"message": "Treniranje je vec u toku"}), 409

    def run_training():
        global model, metadata
        try:
            from train import train as do_train
            m, meta = do_train(epochs=60, batch_size=128, lr=0.001)
            model = m
            metadata = meta
        except Exception as e:
            import traceback
            print(f"Greska pri treniranju: {e}")
            traceback.print_exc()

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    return jsonify({"message": "Treniranje pokrenuto u pozadini!"})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MLP Demo - Prepoznavanje rukopisnih znakova")
    print("=" * 60)
    print("  Ucitavanje modela...")
    if load_model():
        print("  Model uspjesno ucitan!")
    else:
        print("  UPOZORENJE: Model nije pronaden.")
        print("  Pokreni python train.py da treniras model.")
        print("  Ili koristi dugme 'Treniraj' u web interfejsu.")
    print("\n  Server pokrenut na: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
