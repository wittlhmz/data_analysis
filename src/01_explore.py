"""
Schritt 1: Daten explorieren & visualisieren (aus SQLite)
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "database" / "spotify.db"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

print("Lade Daten aus SQLite …")
con = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM songs", con)
con.close()
print(f"  {df.shape[0]:,} Zeilen × {df.shape[1]} Spalten\n")

print("Datentypen:")
print(df.dtypes.to_string())
print(f"\nFehlende Werte: {df.isna().sum().sum()}")

# --- Plot 1: Verteilung der Popularität ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["popularity"], bins=50, color="#185FA5", edgecolor="none")
axes[0].set_title("Verteilung der Popularität")
axes[0].set_xlabel("Popularität (0–100)")
axes[0].set_ylabel("Anzahl Songs")
axes[0].spines[["top", "right"]].set_visible(False)

pop_stats = df["popularity"].describe()
textstr = "\n".join([
    f"Mittelwert:  {pop_stats['mean']:.1f}",
    f"Median:      {pop_stats['50%']:.1f}",
    f"Std:         {pop_stats['std']:.1f}",
    f"Min / Max:   {pop_stats['min']:.0f} / {pop_stats['max']:.0f}",
])
axes[0].text(0.97, 0.97, textstr, transform=axes[0].transAxes,
             fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

top_genres = df["genre"].value_counts().head(10)
top_genres.sort_values().plot(kind="barh", ax=axes[1], color="#185FA5")
axes[1].set_title("Top-10 Genres (Anzahl Songs)")
axes[1].set_xlabel("Anzahl Songs")
axes[1].spines[["top", "right"]].set_visible(False)

fig.suptitle("Spotify-Datensatz: Überblick", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "01_popularity_and_genres.png", dpi=150)
plt.close(fig)
print("Gespeichert: 01_popularity_and_genres.png")

# --- Plot 2: Verteilung der Audio-Features ---
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, feat in enumerate(AUDIO_FEATURES):
    axes[i].hist(df[feat].dropna(), bins=50, color="#1D9E75", edgecolor="none")
    axes[i].set_title(feat)
    axes[i].spines[["top", "right"]].set_visible(False)
    axes[i].set_ylabel("Anzahl")
fig.suptitle("Verteilung der Audio-Features", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "01_audio_feature_distributions.png", dpi=150)
plt.close(fig)
print("Gespeichert: 01_audio_feature_distributions.png")

# --- Plot 3: Korrelationsmatrix ---
corr_cols = AUDIO_FEATURES + ["popularity", "duration_ms", "avg_artist_popularity"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, ax=ax, annot_kws={"size": 8})
ax.set_title("Pearson-Korrelation – Audio-Features & Popularität", fontsize=13, pad=12)
fig.tight_layout()
fig.savefig(OUT / "01_correlation_matrix.png", dpi=150)
plt.close(fig)
print("Gespeichert: 01_correlation_matrix.png")

# --- Plot 4: Popularität nach Genre (Boxplot) ---
top10 = df["genre"].value_counts().head(10).index
df_top = df[df["genre"].isin(top10)]
order = df_top.groupby("genre")["popularity"].median().sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=df_top, x="genre", y="popularity", order=order,
            hue="genre", palette=["#185FA5"] * 10, legend=False,
            ax=ax, fliersize=1.5, linewidth=0.8)
ax.set_title("Popularität nach Genre (Top 10)", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Popularität (0–100)")
ax.tick_params(axis="x", rotation=25)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "01_popularity_by_genre.png", dpi=150)
plt.close(fig)
print("Gespeichert: 01_popularity_by_genre.png")

# --- Plot 5: Durchschnittliche Popularität über die Jahre ---
yearly = df.groupby("year")["popularity"].mean().reset_index()
yearly = yearly[(yearly["year"] >= 1960) & (yearly["year"] <= 2023)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(yearly["year"], yearly["popularity"], color="#185FA5", linewidth=2)
ax.fill_between(yearly["year"], yearly["popularity"], alpha=0.15, color="#185FA5")
ax.set_title("Durchschnittliche Popularität pro Jahr", fontsize=13)
ax.set_xlabel("Jahr")
ax.set_ylabel("Ø Popularität")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "01_popularity_over_years.png", dpi=150)
plt.close(fig)
print("Gespeichert: 01_popularity_over_years.png")

print("\nSchritt 1 abgeschlossen.")
