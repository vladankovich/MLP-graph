"""
Viseslojni perceptron (MLP) - Implementacija od nule koristenjem NumPy-a.
"""

import numpy as np
import pickle
import os


class MLP:
    """
    Multilayer Perceptron (Viseslojni Perceptron).

    Parametri:
        layer_sizes (list): Velicine slojeva [ulaz, skriveni1, ..., izlaz]
        dropout_rates (list): Stopa dropout-a za svaki sloj
        seed (int): Seed za reproduktivnost
    """

    def __init__(self, layer_sizes, dropout_rates=None, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Dropout stope (zadano: 0.3 za skrivene slojeve)
        if dropout_rates is None:
            self.dropout_rates = [0.3] * (len(layer_sizes) - 2) + [0.0]
        else:
            self.dropout_rates = dropout_rates

        # He inicijalizacija tezina (W ~ N(0, sqrt(2/n_in)))
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        # Adam optimizator - stanje prvog i drugog momenta
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_deriv(Z):
        return (Z > 0).astype(np.float64)

    @staticmethod
    def softmax(Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X, training=False):
        self.cache = {"A": [X], "Z": [], "masks": []}
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.cache["Z"].append(Z)
            if i < len(self.weights) - 1:
                A = self.relu(Z)
                if training and self.dropout_rates[i] > 0:
                    p = 1.0 - self.dropout_rates[i]
                    mask = (np.random.random(A.shape) < p) / p
                    A = A * mask
                    self.cache["masks"].append(mask)
                else:
                    self.cache["masks"].append(None)
            else:
                A = self.softmax(Z)
            self.cache["A"].append(A)
        return A

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1.0)
        return -np.sum(y_true * np.log(y_pred)) / m

    def backward(self, y_true):
        m = y_true.shape[0]
        n = len(self.weights)
        dW = [None] * n
        db = [None] * n
        dA = (self.cache["A"][-1] - y_true) / m
        for i in range(n - 1, -1, -1):
            dW[i] = self.cache["A"][i].T @ dA
            db[i] = np.sum(dA, axis=0, keepdims=True)
            if i > 0:
                dA_prev = dA @ self.weights[i].T
                if self.cache["masks"][i - 1] is not None:
                    dA_prev *= self.cache["masks"][i - 1]
                dA = dA_prev * self.relu_deriv(self.cache["Z"][i - 1])
        return dW, db

    def adam_update(self, dW, db, lr):
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        for i in range(len(self.weights)):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * dW[i] ** 2
            self.weights[i] -= lr * (self.m_w[i] / bc1) / (np.sqrt(self.v_w[i] / bc2) + self.epsilon)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * db[i] ** 2
            self.biases[i] -= lr * (self.m_b[i] / bc1) / (np.sqrt(self.v_b[i] / bc2) + self.epsilon)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=50, batch_size=128, lr=0.001, lr_decay=0.97, callback=None):
        n = X_train.shape[0]
        history = []
        print("=" * 70)
        print(f"  Treniranje MLP: {self.layer_sizes}")
        print(f"  Epohe: {epochs} | Batch: {batch_size} | LR: {lr}")
        print("=" * 70)
        for epoch in range(epochs):
            epoch_lr = lr * (lr_decay ** epoch)
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X_train[idx], y_train[idx]
            total_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                y_pred = self.forward(Xb, training=True)
                total_loss += self.cross_entropy_loss(y_pred, yb)
                n_batches += 1
                dW, db = self.backward(yb)
                self.adam_update(dW, db, epoch_lr)
            avg_loss = total_loss / n_batches
            y_tr_pred = self.forward(X_train, training=False)
            train_acc = float(np.mean(np.argmax(y_tr_pred, axis=1) == np.argmax(y_train, axis=1)))
            record = {"epoch": epoch + 1, "loss": float(avg_loss), "train_acc": train_acc, "lr": float(epoch_lr)}
            if X_val is not None:
                y_v_pred = self.forward(X_val, training=False)
                val_acc = float(np.mean(np.argmax(y_v_pred, axis=1) == np.argmax(y_val, axis=1)))
                val_loss = float(self.cross_entropy_loss(y_v_pred, y_val))
                record["val_acc"] = val_acc
                record["val_loss"] = val_loss
                print(f"  Ep {epoch+1:3d}/{epochs} | LR={epoch_lr:.6f} | Loss={avg_loss:.4f} | Train={train_acc*100:.2f}% | Val={val_acc*100:.2f}%")
            else:
                print(f"  Ep {epoch+1:3d}/{epochs} | LR={epoch_lr:.6f} | Loss={avg_loss:.4f} | Train={train_acc*100:.2f}%")
            history.append(record)
            if callback:
                callback(record)
        return history

    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y_onehot):
        return float(np.mean(self.predict(X) == np.argmax(y_onehot, axis=1)))

    def save(self, path):
        data = {
            "layer_sizes": self.layer_sizes,
            "dropout_rates": self.dropout_rates,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Model sacuvan: {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layer_sizes"], data["dropout_rates"])
        model.weights = data["weights"]
        model.biases = data["biases"]
        model.m_w = [np.zeros_like(w) for w in model.weights]
        model.v_w = [np.zeros_like(w) for w in model.weights]
        model.m_b = [np.zeros_like(b) for b in model.biases]
        model.v_b = [np.zeros_like(b) for b in model.biases]
        return model
