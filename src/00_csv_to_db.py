"""
Schritt 0: songs.csv → SQLite-Datenbank
Lyrics-Spalte wird weggelassen (irrelevant für Audio-Analyse, spart ~80% Speicher)
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "data" / "songs.csv"
DB   = ROOT / "database" / "spotify.db"
DB.parent.mkdir(exist_ok=True)

DROP_COLS = ["lyrics", "artist_ids", "niche_genres"]

print(f"Lese {CSV.name} …")
df = pd.read_csv(CSV, low_memory=False)
print(f"  Rohdaten:  {df.shape[0]:,} Zeilen × {df.shape[1]} Spalten")

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"  Nach Drop: {df.shape[0]:,} Zeilen × {df.shape[1]} Spalten")
print(f"  Spalten:   {list(df.columns)}\n")

missing = df.isna().mean().mul(100).round(1)
print("Fehlende Werte (%):")
print(missing[missing > 0].to_string() if missing.any() else "  keine")

if DB.exists():
    DB.unlink()

print(f"\nSchreibe nach {DB} …")
con = sqlite3.connect(DB)
df.to_sql("songs", con, index=False)

print("Erstelle Indizes …")
con.execute("CREATE INDEX idx_genre      ON songs(genre)")
con.execute("CREATE INDEX idx_year       ON songs(year)")
con.execute("CREATE INDEX idx_popularity ON songs(popularity)")
con.commit()

row_check = con.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
con.close()

size_mb = DB.stat().st_size / 1_048_576
print(f"\nDB-Check: {row_check:,} Zeilen in 'songs'")
print(f"Dateigröße: {size_mb:.1f} MB")
print(f"Pfad: {DB}")
print("\nSchritt 0 abgeschlossen.")
