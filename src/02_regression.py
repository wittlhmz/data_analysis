"""
Schritt 2: Regression – Popularität vorhersagen
  Modell A: Lineare Regression  (year + is_rock)
  Modell B: SGD Regressor       (Audio-Features + Artist-Metriken + year + is_rock)
"""

import sqlite3
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "database" / "spotify.db"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# --- Features je Modell ---
FEATURES_LR = ["year", "is_rock"]

FEATURES_SGD = [
    "year",
    "is_rock",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "avg_artist_popularity",
    "total_artist_followers",
]

# --- Daten laden ---
print("Lade Daten …")
con = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM songs", con)
con.close()

df["is_rock"] = (df["genre"] == "Rock").astype(int)
df = df.dropna(subset=FEATURES_SGD + ["popularity"])
print(f"  {len(df):,} Songs nach Bereinigung")

X_all = df[FEATURES_SGD].values
y     = df["popularity"].values

X_lr  = df[FEATURES_LR].values

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=RANDOM_STATE
)
X_train_lr = X_train_all[:, [FEATURES_SGD.index(f) for f in FEATURES_LR]]
X_test_lr  = X_test_all[:,  [FEATURES_SGD.index(f) for f in FEATURES_LR]]

# StandardScaler (für SGD zwingend, für LR zur Vergleichbarkeit)
scaler_lr  = StandardScaler()
scaler_sgd = StandardScaler()
X_train_lr_s  = scaler_lr.fit_transform(X_train_lr)
X_test_lr_s   = scaler_lr.transform(X_test_lr)
X_train_sgd_s = scaler_sgd.fit_transform(X_train_all)
X_test_sgd_s  = scaler_sgd.transform(X_test_all)

# --- Modell A: Lineare Regression ---
print("Trainiere Lineare Regression …")
lr = LinearRegression()
t0 = time.perf_counter()
lr.fit(X_train_lr_s, y_train)
time_lr = time.perf_counter() - t0
y_pred_lr = lr.predict(X_test_lr_s)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)
print(f"  LR   MAE: {mae_lr:.2f}  |  R2: {r2_lr:.4f}  |  Laufzeit: {time_lr:.3f}s")

# --- Modell B: SGD Regressor ---
print("Trainiere SGD Regressor …")
sgd = SGDRegressor(
    loss="huber",
    penalty="elasticnet",
    alpha=0.001,
    l1_ratio=0.15,
    max_iter=1000,
    tol=1e-4,
    random_state=RANDOM_STATE,
)
t0 = time.perf_counter()
sgd.fit(X_train_sgd_s, y_train)
time_sgd = time.perf_counter() - t0
y_pred_sgd = sgd.predict(X_test_sgd_s)
mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
r2_sgd  = r2_score(y_test, y_pred_sgd)
print(f"  SGD  MAE: {mae_sgd:.2f}  |  R2: {r2_sgd:.4f}  |  Laufzeit: {time_sgd:.3f}s")

# --- Plot 1: Modellvergleich ---
models  = ["Lineare Regression\n(year + is_rock)", "SGD Regressor\n(13 Features)"]
maes    = [mae_lr,  mae_sgd]
r2s     = [r2_lr,   r2_sgd]
colors  = ["#185FA5", "#1D9E75"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

bars1 = ax1.bar(models, maes, color=colors, width=0.4)
for bar, val in zip(bars1, maes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_title("Mean Absolute Error (niedriger = besser)")
ax1.set_ylabel("MAE")
ax1.spines[["top", "right"]].set_visible(False)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

bars2 = ax2.bar(models, r2s, color=colors, width=0.4)
for bar, val in zip(bars2, r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_title("R² Score (höher = besser)")
ax2.set_ylabel("R²")
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

fig.suptitle("Regressionsmodelle – Popularität vorhersagen", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "02_model_comparison.png", dpi=150)
plt.close(fig)
print("Gespeichert: 02_model_comparison.png")

# --- Plot 1b: Rock-Songs pro Jahr mit Regressionsgerade ---
rock_per_year = (
    df[df["is_rock"] == 1]
    .groupby("year")
    .size()
    .reset_index(name="count")
)
rock_per_year = rock_per_year[(rock_per_year["year"] >= 1960) & (rock_per_year["year"] <= 2023)]

X_yr = rock_per_year["year"].values.reshape(-1, 1)
y_yr = rock_per_year["count"].values
lr_trend = LinearRegression()
lr_trend.fit(X_yr, y_yr)
y_trend = lr_trend.predict(X_yr)
slope = lr_trend.coef_[0]
r2_trend = r2_score(y_yr, y_trend)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(rock_per_year["year"], rock_per_year["count"],
       color="#185FA5", alpha=0.5, label="Rock-Songs pro Jahr")
ax.plot(rock_per_year["year"], y_trend,
        color="#D85A30", linewidth=2.5,
        label=f"Regressionsgerade  (Steigung: {slope:+.0f} Songs/Jahr, R²={r2_trend:.3f})")
ax.set_xlabel("Jahr")
ax.set_ylabel("Anzahl Rock-Songs")
ax.set_title("Lineare Regression: Anzahl Rock-Songs über die Zeit")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "02_rock_trend.png", dpi=150)
plt.close(fig)
print("Gespeichert: 02_rock_trend.png")

# --- Plot 2: Vorhergesagt vs. Tatsächlich (beide Modelle) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
sample = np.random.default_rng(RANDOM_STATE).integers(0, len(y_test), 3000)

for ax, y_pred, title, color in [
    (ax1, y_pred_lr,  "Lineare Regression (year + is_rock)", "#185FA5"),
    (ax2, y_pred_sgd, "SGD Regressor (13 Features)",         "#1D9E75"),
]:
    ax.scatter(y_test[sample], y_pred[sample], alpha=0.2, s=6,
               color=color, rasterized=True)
    mn, mx = 0, 100
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("Tatsächliche Popularität")
    ax.set_ylabel("Vorhergesagte Popularität")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

fig.suptitle("Vorhergesagt vs. Tatsächlich (3.000 Testpunkte)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "02_predicted_vs_actual.png", dpi=150)
plt.close(fig)
print("Gespeichert: 02_predicted_vs_actual.png")

# --- Plot 3: Feature-Gewichte SGD ---
coef_df = pd.DataFrame({
    "Feature": FEATURES_SGD,
    "Gewicht": sgd.coef_,
}).sort_values("Gewicht", key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_coef = ["#D85A30" if v < 0 else "#1D9E75" for v in coef_df["Gewicht"]]
ax.barh(coef_df["Feature"], coef_df["Gewicht"], color=colors_coef)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("SGD Regressor – Feature-Gewichte", fontsize=13)
ax.set_xlabel("Gewicht (skaliert)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "02_sgd_feature_weights.png", dpi=150)
plt.close(fig)
print("Gespeichert: 02_sgd_feature_weights.png")

# Ergebnisse für README ausgeben
print(f"""
Ergebnisse:
  Lineare Regression   MAE {mae_lr:.2f}  |  R2 {r2_lr:.4f}  |  Laufzeit {time_lr:.3f}s
  SGD Regressor        MAE {mae_sgd:.2f}  |  R2 {r2_sgd:.4f}  |  Laufzeit {time_sgd:.3f}s
""")
print("Schritt 2 abgeschlossen.")
