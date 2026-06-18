"""
Schritt 3: Classification – Genre vorhersagen
  Modell A: LinearSVC
  Modell B: SGD Classifier
  Ziel: Genre aus Audio-Features + Metadaten vorhersagen (Top-8-Genres)
"""

import sqlite3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
TOP_N_GENRES = 8
ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "database" / "spotify.db"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "year", "avg_artist_popularity", "total_artist_followers",
]

# --- Daten laden ---
print("Lade Daten …")
con = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM songs", con)
con.close()

top_genres = df["genre"].value_counts().head(TOP_N_GENRES).index
df = df[df["genre"].isin(top_genres)].dropna(subset=FEATURES + ["genre"])
print(f"  {len(df):,} Songs | {TOP_N_GENRES} Genres")
print(f"  Genres: {list(top_genres)}\n")

le = LabelEncoder()
y = le.fit_transform(df["genre"])
X = df[FEATURES].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- Modell A: LinearSVC ---
print("Trainiere LinearSVC …")
svc = LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
t0 = time.perf_counter()
svc.fit(X_train_s, y_train)
time_svc = time.perf_counter() - t0
y_pred_svc = svc.predict(X_test_s)
acc_svc = accuracy_score(y_test, y_pred_svc)
f1_svc  = f1_score(y_test, y_pred_svc, average="weighted")
print(f"  Accuracy: {acc_svc:.4f}  |  F1: {f1_svc:.4f}  |  Laufzeit: {time_svc:.2f}s")

# --- Modell B: SGD Classifier ---
print("Trainiere SGD Classifier …")
sgd = SGDClassifier(
    loss="modified_huber", penalty="elasticnet",
    class_weight="balanced", max_iter=1000,
    random_state=RANDOM_STATE, n_jobs=-1,
)
t0 = time.perf_counter()
sgd.fit(X_train_s, y_train)
time_sgd = time.perf_counter() - t0
y_pred_sgd = sgd.predict(X_test_s)
acc_sgd = accuracy_score(y_test, y_pred_sgd)
f1_sgd  = f1_score(y_test, y_pred_sgd, average="weighted")
print(f"  Accuracy: {acc_sgd:.4f}  |  F1: {f1_sgd:.4f}  |  Laufzeit: {time_sgd:.2f}s")

genre_labels = le.classes_

# --- Plot 1: Modellvergleich ---
models = [f"LinearSVC", f"SGD Classifier"]
accs   = [acc_svc,  acc_sgd]
f1s    = [f1_svc,   f1_sgd]
colors = ["#185FA5", "#1D9E75"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
for ax, vals, title, ylabel in [
    (ax1, accs, "Accuracy", "Accuracy"),
    (ax2, f1s,  "F1-Score (weighted)", "F1-Score"),
]:
    bars = ax.bar(models, vals, color=colors, width=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, min(1.0, max(vals) * 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

laufzeit_text = f"Laufzeiten: LinearSVC {time_svc:.2f}s  |  SGD {time_sgd:.2f}s"
fig.text(0.5, -0.02, laufzeit_text, ha="center", fontsize=9, color="gray")
fig.suptitle("Klassifikationsmodelle – Genre vorhersagen", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "03_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Gespeichert: 03_model_comparison.png")

# --- Plot 2: Konfusionsmatrizen ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, y_pred, title in [
    (axes[0], y_pred_svc, "LinearSVC"),
    (axes[1], y_pred_sgd, "SGD Classifier"),
]:
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=35)

fig.suptitle("Konfusionsmatrizen (normalisiert) – Genre-Klassifikation", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "03_confusion_matrices.png", dpi=150)
plt.close(fig)
print("Gespeichert: 03_confusion_matrices.png")

# --- Plot 3: F1 pro Genre ---
f1_per_genre_svc = f1_score(y_test, y_pred_svc, average=None)
f1_per_genre_sgd = f1_score(y_test, y_pred_sgd, average=None)

x = np.arange(len(genre_labels))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - w/2, f1_per_genre_svc, w, label="LinearSVC",    color="#185FA5")
ax.bar(x + w/2, f1_per_genre_sgd, w, label="SGD Classifier", color="#1D9E75")
ax.set_xticks(x)
ax.set_xticklabels(genre_labels, rotation=25, ha="right")
ax.set_ylabel("F1-Score")
ax.set_title("F1-Score pro Genre", fontsize=13)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "03_f1_per_genre.png", dpi=150)
plt.close(fig)
print("Gespeichert: 03_f1_per_genre.png")

print(f"""
Ergebnisse:
  LinearSVC       Accuracy {acc_svc:.4f}  |  F1 {f1_svc:.4f}  |  Laufzeit {time_svc:.2f}s
  SGD Classifier  Accuracy {acc_sgd:.4f}  |  F1 {f1_sgd:.4f}  |  Laufzeit {time_sgd:.2f}s
""")
print("Schritt 3 abgeschlossen.")
