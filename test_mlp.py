"""
Automatizovani skup testova za verifikaciju svake komponente i matematičkog modela.
Usklađeno sa završnim radom: "Višeslojni perceptron neuronskih mreža u prepoznavanju slika"
Kandidat: Vladan Kenjić | Mentor: Doc. dr Maid Omerović
FTS Travnik
"""

import unittest
import numpy as np
import os
import json
import base64
import io
from PIL import Image

# Uvoz modula iz tekućeg paketa
from mlp import MLP
from data_loader import load_dataset, one_hot
from app import app, preprocess_canvas_image, center_image


class TestMLPMathematics(unittest.TestCase):
    """Testiranje matematičkih jednačina i komponenti implementiranih u mlp.py."""

    def setUp(self):
        self.layer_sizes = [784, 128, 64, 10]
        self.model = MLP(self.layer_sizes, dropout_rates=[0.2, 0.2, 0.0], seed=42)

    def test_01_parameter_count_exact(self):
        """Provjera tačnog broja slobodnih parametara (Poglavlje V rada, P280-P288)."""
        total, details = self.model.count_parameters()
        # Sloj 1: 784 * 128 + 128 = 100.480
        self.assertEqual(details[0]["weights"], 784 * 128)
        self.assertEqual(details[0]["biases"], 128)
        self.assertEqual(details[0]["total"], 100480)

        # Sloj 2: 128 * 64 + 64 = 8.256
        self.assertEqual(details[1]["weights"], 128 * 64)
        self.assertEqual(details[1]["biases"], 64)
        self.assertEqual(details[1]["total"], 8256)

        # Sloj 3: 64 * 10 + 10 = 650
        self.assertEqual(details[2]["weights"], 64 * 10)
        self.assertEqual(details[2]["biases"], 10)
        self.assertEqual(details[2]["total"], 650)

        # Ukupno: 100.480 + 8.256 + 650 = 109.386
        self.assertEqual(total, 109386, f"Ukupan broj parametara {total} mora biti identičan radu: 109.386!")

    def test_02_relu_activation(self):
        """Provjera nelinearne aktivacije ReLU i njenog prvog izvoda (P174-P178)."""
        z = np.array([[-3.0, 0.0, 2.5, -0.01]], dtype=np.float64)
        a = MLP.relu(z)
        np.testing.assert_array_equal(a, [[0.0, 0.0, 2.5, 0.0]])

        # Izvod
        da = MLP.relu_deriv(z)
        np.testing.assert_array_equal(da, [[0.0, 0.0, 1.0, 0.0]])

    def test_03_softmax_activation(self):
        """Provjera Softmax funkcije i numeričke stabilnosti (P180-P183)."""
        # Test velikih vrednosti (zaštita od overflow-a)
        z = np.array([[1000.0, 1001.0, 1002.0], [-100.0, -100.0, -100.0]], dtype=np.float64)
        sm = MLP.softmax(z)
        # Zbir po vrstama mora biti tačno 1.0
        sums = np.sum(sm, axis=1)
        np.testing.assert_allclose(sums, [1.0, 1.0], rtol=1e-6)
        self.assertTrue(np.all(sm >= 0.0) and np.all(sm <= 1.0))
        # Treći element prve vrste ima najveću vrijednost
        self.assertEqual(np.argmax(sm[0]), 2)

    def test_04_cross_entropy_loss(self):
        """Provjera funkcije gubitka kategoričke unakrsne entropije (P188-P192)."""
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        # Savršena predikcija
        y_pred_perfect = np.array([[0.99999, 0.00001], [0.00001, 0.99999]], dtype=np.float64)
        loss_perfect = self.model.cross_entropy_loss(y_pred_perfect, y_true)
        self.assertLess(loss_perfect, 0.001)

        # Pogrešna predikcija
        y_pred_bad = np.array([[0.01, 0.99], [0.99, 0.01]], dtype=np.float64)
        loss_bad = self.model.cross_entropy_loss(y_pred_bad, y_true)
        self.assertGreater(loss_bad, 4.0)

    def test_05_forward_and_backward_shapes(self):
        """Provjera dimenzija matrica u unaprijednom i unazadnom prolazu (P149-P210)."""
        batch_size = 16
        X = np.random.randn(batch_size, 784).astype(np.float64)
        y = np.zeros((batch_size, 10), dtype=np.float64)
        y[:, 3] = 1.0

        # Unaprijedni prolaz
        y_pred = self.model.forward(X, training=True)
        self.assertEqual(y_pred.shape, (batch_size, 10))
        self.assertEqual(len(self.model.cache["A"]), 4)  # Ulaz + 2 skrivena + Izlaz
        self.assertEqual(self.model.cache["A"][0].shape, (batch_size, 784))
        self.assertEqual(self.model.cache["A"][1].shape, (batch_size, 128))
        self.assertEqual(self.model.cache["A"][2].shape, (batch_size, 64))
        self.assertEqual(self.model.cache["A"][3].shape, (batch_size, 10))

        # Propagacija unazad
        dW, db = self.model.backward(y)
        self.assertEqual(len(dW), 3)
        self.assertEqual(len(db), 3)

        self.assertEqual(dW[0].shape, (784, 128))
        self.assertEqual(db[0].shape, (1, 128))
        self.assertEqual(dW[1].shape, (128, 64))
        self.assertEqual(db[1].shape, (1, 64))
        self.assertEqual(dW[2].shape, (64, 10))
        self.assertEqual(db[2].shape, (1, 10))

    def test_06_adam_optimizer_update(self):
        """Provjera ažuriranja težina i pomjeraja kroz Adam algoritam (P214)."""
        X = np.random.randn(8, 784).astype(np.float64)
        y = np.zeros((8, 10), dtype=np.float64)
        y[:, 1] = 1.0

        self.model.forward(X, training=True)
        dW, db = self.model.backward(y)

        # Čuvanje početnih težina
        w0_before = self.model.weights[0].copy()
        b0_before = self.model.biases[0].copy()

        self.model.adam_update(dW, db, lr=0.001)

        # Težine i pomjeraji moraju biti promijenjeni
        self.assertFalse(np.array_equal(w0_before, self.model.weights[0]))
        self.assertFalse(np.array_equal(b0_before, self.model.biases[0]))
        self.assertEqual(self.model.t, 1)

    def test_07_confusion_matrix_computation(self):
        """Provjera računanja matrice konfuzije (Tabela 1 rada)."""
        X = np.random.randn(20, 784).astype(np.float64)
        y = np.random.randint(0, 10, size=20)
        cm = self.model.compute_confusion_matrix(X, y)
        self.assertEqual(cm.shape, (10, 10))
        self.assertEqual(np.sum(cm), 20)

    def test_08_save_and_load(self):
        """Provjera ispravnosti serijalizacije i učitavanja modela."""
        temp_path = "models/test_temp_model.pkl"
        os.makedirs("models", exist_ok=True)
        self.model.save(temp_path)

        loaded = MLP.load(temp_path)
        self.assertEqual(loaded.layer_sizes, self.model.layer_sizes)
        np.testing.assert_array_equal(loaded.weights[0], self.model.weights[0])
        np.testing.assert_array_equal(loaded.biases[0], self.model.biases[0])

        if os.path.exists(temp_path):
            os.remove(temp_path)


class TestDataLoader(unittest.TestCase):
    """Testiranje modula za pripremu podataka."""

    def test_09_one_hot_encoding(self):
        """Provjera one-hot transformacije oznaka klasa."""
        y = np.array([0, 3, 9, 7])
        oh = one_hot(y, n_classes=10)
        self.assertEqual(oh.shape, (4, 10))
        self.assertEqual(oh[0, 0], 1.0)
        self.assertEqual(oh[1, 3], 1.0)
        self.assertEqual(oh[2, 9], 1.0)
        self.assertEqual(oh[3, 7], 1.0)
        self.assertEqual(np.sum(oh), 4.0)

    def test_10_mnist_shapes_and_ranges(self):
        """Provjera normalizacije i dimenzija MNIST skupa."""
        X_tr, y_tr, X_te, y_te, n_classes, class_names = load_dataset()
        self.assertEqual(X_tr.shape, (60000, 784))
        self.assertEqual(X_te.shape, (10000, 784))
        self.assertEqual(n_classes, 10)
        self.assertEqual(class_names, [str(i) for i in range(10)])
        self.assertGreaterEqual(float(X_tr.min()), 0.0)
        self.assertLessEqual(float(X_tr.max()), 1.0)


class TestAppAndPreprocessing(unittest.TestCase):
    """Testiranje predobrade slika i Flask REST API ruta."""

    def setUp(self):
        self.app = app.test_client()

    def test_11_image_preprocessing_pipeline(self):
        """Provjera skaliranja i linearizacije slike sa Canvas-a."""
        # Kreiranje testne PNG slike (bela linija na crnoj pozadini)
        img = Image.new("RGBA", (280, 280), (0, 0, 0, 255))
        for x in range(100, 180):
            for y in range(100, 180):
                img.putpixel((x, y), (255, 255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        processed = preprocess_canvas_image(b64)
        self.assertEqual(processed.shape, (1, 784))
        self.assertGreater(float(processed.max()), 0.5)
        self.assertLessEqual(float(processed.max()), 1.0)
        self.assertGreaterEqual(float(processed.min()), 0.0)

    def test_12_flask_endpoints(self):
        """Provjera osnovnih Flask endpointa."""
        # Status ruta
        res_status = self.app.get("/api/status")
        self.assertIn(res_status.status_code, [200, 503])

        # Glavna stranica
        res_index = self.app.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"MLP Demo 1.0", res_index.data)


if __name__ == "__main__":
    unittest.main()
