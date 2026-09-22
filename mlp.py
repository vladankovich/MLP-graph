"""
Višeslojni perceptron (MLP) — Multilayer Perceptron
Implementacija od nule (from scratch) korišćenjem biblioteke NumPy.
Usklađeno sa završnim (diplomskim) radom:
"Višeslojni perceptron neuronskih mreža u prepoznavanju slika"
Kandidat: Vladan Kenjić | Mentor: Doc. dr Maid Omerović
Fakultet za tehničke studije (FTS), Univerzitet u Travniku
"""

import numpy as np
import pickle
import os


class MLP:
    """
    Višeslojni perceptron (Multilayer Perceptron — MLP).
    
    Projektovana arhitektura za MNIST klasifikaciju:
        - Ulazni sloj: 784 neurona (linearizovana matrica slike 28x28 piksela)
        - Prvi skriveni sloj: 128 neurona (ReLU aktivacija, 100.480 parametara)
        - Drugi skriveni sloj: 64 neurona (ReLU aktivacija, 8.256 parametara)
        - Izlazni sloj: 10 neurona (Softmax aktivacija, 650 parametara)
        - Ukupan broj slobodnih (obučivih) parametara: 109.386
    
    Parametri:
        layer_sizes (list): Broj neurona po slojevima [784, 128, 64, 10]
        dropout_rates (list): Stopa isključivanja neurona za skrivene slojeve [0.2, 0.2, 0.0]
        seed (int): Seme za generator slučajnih brojeva radi reproduktivnosti
    """

    def __init__(self, layer_sizes=None, dropout_rates=None, seed=42):
        if layer_sizes is None:
            # Podrazumijevana arhitektura definisana u radu (Poglavlje V, P282-P288)
            layer_sizes = [784, 128, 64, 10]

        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Regularizacija isključivanjem neurona (Dropout)
        # Za skrivene slojeve koristi se stopa p (20%), na izlaznom sloju nema dropout-a
        if dropout_rates is None:
            self.dropout_rates = [0.2] * (len(layer_sizes) - 2) + [0.0]
        else:
            self.dropout_rates = dropout_rates

        # Inicijalizacija težina: He (Kaiming) metod optimalan za ReLU aktivaciju
        # W ~ N(0, sqrt(2 / n_in))
        # Vektori pomjeraja (biases) inicijalizuju se na nulu: b = 0
        self.weights = []  # Matrice težina W^[l]
        self.biases = []   # Vektori pomjeraja b^[l]

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            W = np.random.randn(n_in, n_out).astype(np.float64) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out), dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

        # Optimizator Adam (Adaptive Moment Estimation)
        # Stanja prvog momenta (m - srednja vrijednost) i drugog momenta (v - varijansa)
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        # Hiperparametri Adam optimizatora prema literaturi (Kingma & Ba, 2014)
        self.t = 0              # Vremenski korak (iteracija)
        self.beta1 = 0.9        # Faktor eksponencijalnog opadanja prvog momenta
        self.beta2 = 0.999      # Faktor eksponencijalnog opadanja drugog momenta
        self.epsilon = 1e-8     # Mala konstanta radi sprečavanja deljenja nulom

        # Keš za vrijednosti tokom unaprijednog prostiranja (forward pass)
        self.cache = {}

    def count_parameters(self):
        """
        Računa tačan broj obučivih parametara po slojevima i ukupno.
        Za arhitekturu [784, 128, 64, 10]:
            Sloj 1: 784 * 128 + 128 = 100.480
            Sloj 2: 128 * 64 + 64   = 8.256
            Sloj 3: 64 * 10 + 10    = 650
            Ukupno: 109.386 parametara.
        """
        details = []
        total = 0
        for i in range(len(self.weights)):
            w_params = self.weights[i].size
            b_params = self.biases[i].size
            layer_params = w_params + b_params
            total += layer_params
            details.append({
                "layer": i + 1,
                "n_in": self.layer_sizes[i],
                "n_out": self.layer_sizes[i + 1],
                "weights": w_params,
                "biases": b_params,
                "total": layer_params
            })
        return total, details

    # =========================================================================
    # Nelinearne aktivacione funkcije (Poglavlje III, P159-P183)
    # =========================================================================

    @staticmethod
    def relu(Z):
        """
        Ispravljena linearna jedinica (Rectified Linear Unit — ReLU):
        phi(z) = max(0, z)
        Otklanja problem iščezavajućeg gradijenta u skrivenim slojevima.
        """
        return np.maximum(0.0, Z)

    @staticmethod
    def relu_deriv(Z):
        """
        Prvi izvod ReLU aktivacione funkcije:
        phi'(z) = 1 ako je z > 0, inače 0.
        """
        return (Z > 0.0).astype(np.float64)

    @staticmethod
    def softmax(Z):
        """
        Softmax funkcija aktivacije na izlaznom sloju:
        y_hat_k = exp(z_k) / sum(exp(z_j))
        Numerički stabilna implementacija oduzimanjem maksimalne vrijednosti po vrstama.
        Preslikava logite u normalizovanu raspodjelu vjerovatnoća čiji je zbir 1.
        """
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # =========================================================================
    # Funkcija gubitka (Poglavlje III, P184-P193)
    # =========================================================================

    def cross_entropy_loss(self, y_pred, y_true):
        """
        Funkcija gubitka kategoričke unakrsne entropije (Categorical Cross-Entropy):
        J(W, b) = - (1 / M) * sum_m sum_k ( y_k^(m) * log(y_hat_k^(m)) )
        Optimalna funkcija gubitka za višeklasnu klasifikaciju u kombinaciji sa Softmax izlazom.
        """
        m = y_true.shape[0]
        # Ograničavanje radi sprečavanja numeričkih problema log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.sum(y_true * np.log(y_pred_clipped)) / m

    # =========================================================================
    # Unaprijedno prostiranje signala (Forward Pass) (Poglavlje III, P149-P158)
    # =========================================================================

    def forward(self, X, training=False):
        """
        Unaprijedno prostiranje signala kroz sve slojeve mreže:
            z^[l] = a^[l-1] * W^[l] + b^[l]       (linearna transformacija)
            a^[l] = phi^[l](z^[l])                 (nelinearna aktivacija)
        
        Tokom treninga primjenjuje se invertovani Dropout (Inverted Dropout):
            a^[l] = (a^[l] * mask) / p,  gdje je p = 1 - dropout_rate
        """
        self.cache = {
            "A": [X],     # Vektori aktivacija: A^[0] = X
            "Z": [],      # Ponderisane sume / indukovana lokalna polja Z^[l]
            "masks": []   # Dropout maske
        }
        A = X
        n_layers_weights = len(self.weights)

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Linearna transformacija: Z = A * W + b
            Z = A @ W + b
            self.cache["Z"].append(Z)

            if i < n_layers_weights - 1:
                # Skriveni slojevi: ReLU aktivacija
                A = self.relu(Z)

                # Invertovani dropout tokom obučavanja
                if training and self.dropout_rates[i] > 0.0:
                    keep_prob = 1.0 - self.dropout_rates[i]
                    mask = (np.random.random(A.shape) < keep_prob) / keep_prob
                    A = A * mask
                    self.cache["masks"].append(mask)
                else:
                    self.cache["masks"].append(None)
            else:
                # Izlazni sloj: Softmax aktivacija (vjerovatnoće klasa)
                A = self.softmax(Z)

            self.cache["A"].append(A)

        return A

    # =========================================================================
    # Propagacija greške unazad (Backpropagation) (Poglavlje III, P194-P210)
    # =========================================================================

    def backward(self, y_true):
        """
        Izračunavanje parcijalnih izvoda funkcije greške primjenom lančanog pravila.
        
        Izlazni lokalni gradijent (signal greške za Softmax + Cross-Entropy):
            delta^[L] = (a^[L] - y) / M
        
        Gradijenti parametara sloja l:
            dW^[l] = (a^[l-1])^T * delta^[l]
            db^[l] = sum(delta^[l], axis=0, keepdims=True)
        
        Propagacija signala greške u prethodni skriveni sloj:
            delta^[l-1] = (delta^[l] * (W^[l])^T) * phi'(z^[l-1]) [* mask^[l-1]]
        """
        m = y_true.shape[0]
        n = len(self.weights)
        dW = [None] * n
        db = [None] * n

        # Signal greške na izlaznom sloju: delta^[L] = (y_pred - y_true) / m
        delta = (self.cache["A"][-1] - y_true) / m

        for i in range(n - 1, -1, -1):
            # Gradijent matrice težina dW^[l] i vektora pomjeraja db^[l]
            dW[i] = self.cache["A"][i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagacija signala greške unazad kroz težinske veze
                delta_prev = delta @ self.weights[i].T

                # Propagacija kroz dropout masku (ako je postojala)
                if self.cache["masks"][i - 1] is not None:
                    delta_prev *= self.cache["masks"][i - 1]

                # Množenje sa izvodom nelinearne aktivacione funkcije skrivenog sloja (ReLU')
                delta = delta_prev * self.relu_deriv(self.cache["Z"][i - 1])

        return dW, db

    # =========================================================================
    # Adam optimizator — Ažuriranje težina i pomjeraja (Poglavlje III, P211-P214)
    # =========================================================================

    def adam_update(self, dW, db, lr):
        """
        Ažuriranje parametara primjenom Adam algoritma optimizacije.
        
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (prvi momenat)
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (drugi momenat)
        m_hat = m_t / (1 - beta1^t)                        (korekcija pristrasnosti)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
        """
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i in range(len(self.weights)):
            # Ažuriranje prvog i drugog momenta za težine
            self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * dW[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * (dW[i] ** 2)
            m_w_hat = self.m_w[i] / bc1
            v_w_hat = self.v_w[i] / bc2
            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

            # Ažuriranje prvog i drugog momenta za pomjeraje (biases)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * db[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (db[i] ** 2)
            m_b_hat = self.m_b[i] / bc1
            v_b_hat = self.v_b[i] / bc2
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    # =========================================================================
    # Obučavanje mreže (Fit metoda) (Poglavlje V i VI, P289-P296)
    # =========================================================================

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=40, batch_size=64, lr=0.001, lr_decay=0.97, callback=None):
        """
        Glavna petlja za obučavanje višeslojnog perceptrona.
        
        Parametri prema radu (P289):
            epochs = 40 (ukupno 40 epoha obučavanja)
            batch_size = 64 (mini-grupe od po 64 uzorka)
            lr = 0.001 (početna stopa učenja eta = 0.001)
        """
        n = X_train.shape[0]
        history = []
        total_params, _ = self.count_parameters()

        print("=" * 75)
        print(f"  TRENIRANJE MLP MREŽE — Arhitektura: {self.layer_sizes}")
        print(f"  Ukupan broj parametara: {total_params:,} (100% usklađeno sa radom)")
        print(f"  Epohe: {epochs} | Veličina paketa (batch): {batch_size} | Stopa učenja: {lr}")
        print("=" * 75)

        for epoch in range(epochs):
            # Eksponencijalno opadanje stope učenja radi fine konvergencije
            epoch_lr = lr * (lr_decay ** epoch)

            # Slučajno mešanje podataka na početku svake epohe (Shuffle)
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            total_loss, n_batches = 0.0, 0

            # Iteracija po mini-paketima (Mini-batch SGD / Adam)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]

                # 1. Unaprijedno prostiranje signala sa aktivnim dropout-om
                y_pred = self.forward(Xb, training=True)
                total_loss += self.cross_entropy_loss(y_pred, yb)
                n_batches += 1

                # 2. Propagacija signala greške unazad
                dW, db = self.backward(yb)

                # 3. Ažuriranje težina i pomjeraja Adam optimizatorom
                self.adam_update(dW, db, epoch_lr)

            avg_loss = total_loss / n_batches

            # Evaluacija na celom trening skupu (bez dropout-a)
            y_tr_pred = self.forward(X_train, training=False)
            train_acc = float(np.mean(np.argmax(y_tr_pred, axis=1) == np.argmax(y_train, axis=1)))

            record = {
                "epoch": epoch + 1,
                "loss": float(avg_loss),
                "train_acc": train_acc,
                "lr": float(epoch_lr)
            }

            # Evaluacija na validacionom / testnom skupu (bez dropout-a)
            if X_val is not None and y_val is not None:
                y_v_pred = self.forward(X_val, training=False)
                val_acc = float(np.mean(np.argmax(y_v_pred, axis=1) == np.argmax(y_val, axis=1)))
                val_loss = float(self.cross_entropy_loss(y_v_pred, y_val))
                record["val_acc"] = val_acc
                record["val_loss"] = val_loss
                print(f"  Epoha {epoch+1:2d}/{epochs:2d} | LR={epoch_lr:.6f} | Gubitak={avg_loss:.4f} | Trening Tačnost={train_acc*100:.2f}% | Test Tačnost={val_acc*100:.2f}%")
            else:
                print(f"  Epoha {epoch+1:2d}/{epochs:2d} | LR={epoch_lr:.6f} | Gubitak={avg_loss:.4f} | Trening Tačnost={train_acc*100:.2f}%")

            history.append(record)
            if callback:
                callback(record)

        return history

    # =========================================================================
    # Inferencija i evaluacija (Poglavlje VI, P297-P307)
    # =========================================================================

    def predict_proba(self, X):
        """Vraća normalizovani vektor vjerovatnoća klasa (Softmax)."""
        return self.forward(X, training=False)

    def predict(self, X):
        """Vraća indeks predviđene klase sa maksimalnom vjerovatnoćom."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y_onehot):
        """Računa tačnost klasifikacije na zadatom skupu."""
        y_true = np.argmax(y_onehot, axis=1) if y_onehot.ndim > 1 else y_onehot
        return float(np.mean(self.predict(X) == y_true))

    def compute_confusion_matrix(self, X, y_true):
        """
        Računa matricu konfuzije dimenzija K x K (Tabela 1 u radu).
        Vrste predstavljaju stvarne klase (S_i), a kolone predviđene (P_j).
        """
        if y_true.ndim > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true

        y_pred_labels = self.predict(X)
        n_classes = self.layer_sizes[-1]
        cm = np.zeros((n_classes, n_classes), dtype=int)

        for true_l, pred_l in zip(y_true_labels, y_pred_labels):
            cm[true_l, pred_l] += 1

        return cm

    # =========================================================================
    # Serijalizacija i perzistencija modela
    # =========================================================================

    def save(self, path):
        """Čuvanje modela sa svim težinama, pomjerajima i konfiguracijom."""
        data = {
            "layer_sizes": self.layer_sizes,
            "dropout_rates": self.dropout_rates,
            "weights": self.weights,
            "biases": self.biases,
            "t": self.t,
            "architecture_note": "MLP 784-128-64-10 (109.386 obučivih parametara)"
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Model uspješno sačuvan u: {path}")

    @classmethod
    def load(cls, path):
        """Učitavanje sačuvanog modela."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layer_sizes"], data["dropout_rates"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        model.t = data.get("t", 0)
        model.m_w = [np.zeros_like(w) for w in model.weights]
        model.v_w = [np.zeros_like(w) for w in model.weights]
        model.m_b = [np.zeros_like(b) for b in model.biases]
        model.v_b = [np.zeros_like(b) for b in model.biases]
        return model
