"""Replica didactica del articulo 17 con el dataset publico de Kaggle."""

from pathlib import Path
import json
import os
import random
import urllib.request
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "dataset.mat"
OUT = ROOT / "resultados"
CLASSES = ["normal", "inner", "outer", "roller"]
WINDOW = 1000
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/julienjta/engine-acoustic-emissions"


def ensure_dataset():
    """Descarga y extrae el dataset público cuando no existe localmente."""
    if DATASET.exists():
        return
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    archive = DATASET.parent / "engine-acoustic-emissions.zip"
    print("Descargando Engine Acoustic Emissions desde Kaggle...")
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extract("dataset.mat", DATASET.parent)
    archive.unlink(missing_ok=True)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def windows(signal, size=WINDOW):
    n = len(signal) // size
    x = signal[: n * size].reshape(n, size, 1).astype("float32")
    return x


def load_splits():
    raw = loadmat(DATASET)
    xs = {"train": [], "val": [], "test": []}
    ys = {"train": [], "val": [], "test": []}
    inventory = {}
    for label, name in enumerate(CLASSES):
        signal = np.asarray(raw[name]).squeeze()
        inventory[name] = int(signal.size)
        a, b = int(signal.size * 0.70), int(signal.size * 0.85)
        pieces = {"train": signal[:a], "val": signal[a:b], "test": signal[b:]}
        for split, piece in pieces.items():
            x = windows(piece)
            xs[split].append(x)
            ys[split].append(np.full(len(x), label, dtype="int64"))
    result = {}
    for split in xs:
        x, y = np.concatenate(xs[split]), np.concatenate(ys[split])
        idx = np.random.default_rng(SEED).permutation(len(x))
        result[split] = (x[idx], y[idx])
    return result, inventory


def build_model(kind):
    recurrent = {"RNN": tf.keras.layers.SimpleRNN, "GRU": tf.keras.layers.GRU,
                 "LSTM_AdamW": tf.keras.layers.LSTM}[kind]
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(WINDOW, 1)),
        recurrent(64),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    optimizer = (tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-4)
                 if kind == "LSTM_AdamW" else tf.keras.optimizers.Adam(1e-3))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    seed_all()
    ensure_dataset()
    OUT.mkdir(parents=True, exist_ok=True)
    splits, inventory = load_splits()
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]
    # Normalizacion global calculada solamente con entrenamiento. Conserva las
    # diferencias relativas entre clases y evita usar informacion de prueba.
    mean, std = float(x_train.mean()), float(x_train.std() + 1e-8)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    metrics, histories = [], {}
    for kind in ["RNN", "GRU", "LSTM_AdamW"]:
        seed_all()
        model = build_model(kind)
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32,
            verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True)])
        pred = model.predict(x_test, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        metrics.append({"modelo": kind, "exactitud": acc, "f1_macro": f1,
                        "epocas": len(history.history["loss"])})
        histories[kind] = history.history
        cm = confusion_matrix(y_test, pred, labels=range(4))
        plt.figure(figsize=(6.2, 5.0))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title(f"Matriz de confusion - {kind}")
        plt.xlabel("Prediccion")
        plt.ylabel("Clase real")
        plt.tight_layout()
        plt.savefig(OUT / f"matriz_confusion_{kind}.png", dpi=180)
        plt.close()
        (OUT / f"reporte_{kind}.txt").write_text(
            classification_report(y_test, pred, target_names=CLASSES, digits=4), encoding="utf-8")
        model.save(OUT / f"modelo_{kind}.keras")
    pd.DataFrame(metrics).to_csv(OUT / "metricas_modelos.csv", index=False)
    plt.figure(figsize=(8, 4.5))
    for kind, hist in histories.items():
        plt.plot(hist["val_accuracy"], label=kind)
    plt.title("Exactitud de validacion")
    plt.xlabel("Epoca")
    plt.ylabel("Exactitud")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "curvas_validacion.png", dpi=180)
    plt.close()
    summary = {
        "dataset": "Engine Acoustic Emissions (Kaggle)",
        "muestras_por_clase": inventory,
        "ventana": WINDOW,
        "division": "70/15/15 cronologica antes de segmentar",
        "segmentos": {k: int(len(v[0])) for k, v in splits.items()},
        "resultados": metrics,
        "advertencia": "Una señal continua por clase; validacion preliminar, no industrial."
    }
    (OUT / "resumen_experimento.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(json.dumps(summary["segmentos"], indent=2))


if __name__ == "__main__":
    main()
