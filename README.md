# Spotify Song Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-database-lightgrey?logo=sqlite)
![Status](https://img.shields.io/badge/Status-abgeschlossen-brightgreen)

Dieses Projekt analysiert einen Spotify-Datensatz mit **550.622 Songs** und ihren Audio-Features mithilfe von Python und scikit-learn.
Ziel ist es, dass ich lerne, wie man auch mit großen Datensätzen umgeht, aber auch wie man Ergebnisse interpretiert. Bei meinem ersten Projekt (https://github.com/wittlhmz/audio_analysis) hatte ich einen sehr kleinen Datensatz und zudem waren die Erkenntnisse nicht besonders spannend.

---

## Datensatz

| Eigenschaft | Wert |
|---|---|
| Quelle | Spotify Web API (via Kaggle) |
| Songs | 550.622 |
| Features | 21 Spalten |
| Zeitraum | 1900 – 2025 |
| Fehlende Werte | < 0,01 % |

**Audio-Features:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`  
**Metadaten:** `name`, `artists`, `album_name`, `genre`, `year`, `popularity`, `total_artist_followers`, `avg_artist_popularity`

Zunächst habe ich bei diesem Datensatz die `lyrics`-Spalte weggelassen, da die Songtexte für meine Analyse irrelevant waren und die Dateigröße der Datenbank damit um ca. 80% reduziert wurde.

---

## Exploration & Visualisierung

Bei einem so großen Datensatz macht es Sinn, dass man sich erstmal mit dem Datensatz vertraut macht und ein paar Verteilungen visualisiert, im Folgenden sind einige mehr oder weniger interessante Fakten visualisiert.

### Popularitätsverteilung & Top-Genres

<p align="center">
  <img src="outputs/01_popularity_and_genres.png" width="800"/>
</p>

Die Popularität ist stark rechtsschief verteilt: Der Mittelwert liegt bei 17,6, der Median bei nur 14 – und 27 % aller Songs haben einen Wert von exakt 0. Das spiegelt die Realität auf Spotify wider, wo nur ein kleiner Bruchteil der Tracks nennenswerte Reichweite erzielt. Rock dominiert den Datensatz mit knapp 197.000 Einträgen und ist damit doppelt so häufig vertreten wie Pop (72.500). Die starke Genre-Ungleichverteilung ist ein wichtiger Faktor, den Klassifikationsmodelle in Schritt 3 berücksichtigen müssen.

---

### Verteilung der Audio-Features

<p align="center">
  <img src="outputs/01_audio_feature_distributions.png" width="800"/>
</p>

Die meisten Audio-Features bewegen sich im normierten Bereich von 0 bis 1. `Instrumentalness` und `speechiness` sind extrem rechtsskewed – die große Mehrheit der Songs hat wenig Instrumentalpassagen und kaum gesprochenes Wort. `Danceability`, `energy` und `valence` folgen annähernd einer Normalverteilung, was sie zu stabilen Features für Machine-Learning-Modelle macht. `Loudness` liegt typischerweise zwischen −20 und 0 dB und zeigt, dass moderne Produktionstechniken Songs deutlich lauter als ältere Aufnahmen machen.

---

### Korrelationsmatrix

<p align="center">
  <img src="outputs/01_correlation_matrix.png" width="800"/>
</p>

Die stärkste Korrelation im Datensatz besteht zwischen `energy` und `loudness` (r = 0,78): lautere Songs werden konsistent als energiereicher eingestuft. Ebenso stark ist der negative Zusammenhang zwischen `acousticness` und `energy` (r = −0,75) – akustische Songs sind naturgemäß ruhiger und weniger intensiv. Für die Popularitätsvorhersage ist `avg_artist_popularity` mit r = 0,39 der stärkste Prädiktor, gefolgt von `total_artist_followers` (r = 0,23) und `danceability` (r = 0,13). Reine Audio-Features haben also einen geringeren Einfluss auf den Erfolg als der bereits etablierte Bekanntheitsgrad des Künstlers.

---

### Popularität nach Genre

<p align="center">
  <img src="outputs/01_popularity_by_genre.png" width="800"/>
</p>

Die Boxplots zeigen deutliche Unterschiede in der Popularitätsverteilung zwischen den Genres. Pop und Hip-Hop tendieren zu höheren Medianwerten, während Jazz, Classical und Blues viele Songs mit Popularität nahe 0 aufweisen. Die Interquartilsabstände sind in allen Genres groß – selbst innerhalb eines Genres gibt es sowohl absolute Hits als auch vollständig ungehörte Tracks. Das macht Genre allein zu einem schwachen Prädiktor für Popularität.

---

## Regression

In meinem anderen Datenprojekt hätte es kaum Sinn gemacht auf die Regression einzugehen, weil die Daten sehr wenige Features hatten, die dann zum Teil auch wenig Aussagekraft haben.

| Modell | Features | MAE | R² | Laufzeit |
|---|---|---|---|---|
| Lineare Regression | `year`, `is_rock` | 14,56 | 0,0095 | 0,030 s |
| SGD Regressor | 13 Features (Audio + Artist) | 12,79 | 0,1663 | 2,798 s |

### Lineare Regression – Rock-Songs über die Zeit

<p align="center">
  <img src="outputs/02_rock_trend.png" width="800"/>
</p>

Die Regressionsgerade zeigt, dass die Anzahl an Rock-Songs im Datensatz über die Jahrzehnte stark zugenommen hat – von wenigen hundert Einträgen in den 1960ern bis zu mehreren tausend pro Jahr ab den 2000ern. Das R² der Trendlinie liegt bei 0,72, was auf einen klar linearen Wachstumstrend hindeutet. Ab etwa 2010 flacht die Kurve leicht ab, was auf eine Sättigung im Rock-Segment oder eine veränderte Genrestruktur auf Spotify hindeuten kann. Die Steigung der Geraden lässt sich direkt aus dem Regressionskoeffizienten ablesen: Pro Jahr kommen im Schnitt mehrere hundert Rock-Songs hinzu.

---

### SGD Regressor – Vorhergesagt vs. Tatsächlich

<p align="center">
  <img src="outputs/02_predicted_vs_actual.png" width="800"/>
</p>

Der SGD Regressor nutzt 13 Features – alle Audio-Merkmale sowie `avg_artist_popularity` und `total_artist_followers` – um die Popularität vorherzusagen. Im Streuplot zeigt sich, dass die Vorhersagen zur Mitte des Wertebereichs tendieren: sehr populäre Songs werden systematisch unterschätzt, Songs mit Popularität 0 überschätzt. Das liegt an der stark schiefen Zielverteilung – 27 % der Songs haben Popularität 0. Mit einem R² von 0,167 erklärt das Modell immerhin 17 % der Varianz, was für eine reine Audio-Feature-Basis beachtlich ist.

---

### SGD Regressor – Feature-Gewichte

<p align="center">
  <img src="outputs/02_sgd_feature_weights.png" width="800"/>
</p>

Die Feature-Gewichte bestätigen die Korrelationsanalyse aus der Exploration: `avg_artist_popularity` hat mit Abstand den größten positiven Einfluss auf die vorhergesagte Popularität – der Bekanntheitsgrad des Künstlers schlägt alle Audio-Merkmale. `instrumentalness` wirkt sich negativ aus, rein instrumentale Songs erzielen im Schnitt geringere Popularitätswerte. Audio-Features wie `danceability`, `energy` und `valence` haben moderate positive Gewichte und liefern dennoch einen messbaren Beitrag zur Vorhersage.

---

## Classification

Ziel: Das **Genre** eines Songs aus seinen Audio-Features vorhersagen. Verwendet werden die Top-8-Genres (522.473 Songs). Da Rock mit ~38 % stark überrepräsentiert ist, wird `class_weight="balanced"` eingesetzt.

| Modell | Accuracy | F1 (weighted) | Laufzeit |
|---|---|---|---|
| LinearSVC | 0,4852 | 0,4422 | 15,18 s |
| SGD Classifier | 0,4685 | 0,4378 | 3,80 s |

### LinearSVC – Modellvergleich

<p align="center">
  <img src="outputs/03_model_comparison.png" width="800"/>
</p>

Beide Modelle erreichen eine Accuracy von knapp unter 50 % – deutlich über dem Zufallsniveau (12,5 % bei 8 gleichverteilten Klassen), aber ein klares Zeichen, dass Genre aus Audio-Features allein schwer zu unterscheiden ist. LinearSVC übertrifft den SGD Classifier leicht bei Accuracy (48,5 % vs. 46,9 %), benötigt dafür aber mit 15,2 s rund viermal so lang. Der SGD Classifier ist mit 3,8 s die deutlich effizientere Wahl, wenn Geschwindigkeit wichtiger ist als maximale Genauigkeit.

---

### Konfusionsmatrizen

<p align="center">
  <img src="outputs/03_confusion_matrices.png" width="800"/>
</p>

Die Konfusionsmatrizen zeigen, wo die Modelle systematisch scheitern: Rock wird am häufigsten mit Folk und Country verwechselt, was musikalisch naheliegend ist – diese Genres teilen oft ähnliche Tempo- und Akustik-Werte. Electronic wird dagegen vergleichsweise gut erkannt, da es sich durch hohe Energie und niedrige Acousticness klar abgrenzt. R&B und Hip-Hop werden häufig gegenseitig verwechselt, was auf ihre Audio-Feature-Ähnlichkeit hindeutet. Jazz wird von beiden Modellen am schlechtesten klassifiziert und oft als Folk oder Country fehlklassifiziert.

---

### F1-Score pro Genre

<p align="center">
  <img src="outputs/03_f1_per_genre.png" width="800"/>
</p>

Der genreweise F1-Score macht die Unterschiede noch klarer: Electronic erzielt den höchsten F1 bei beiden Modellen, da seine Audio-Charakteristik einzigartig ist. Rock hat trotz starker Präsenz im Datensatz einen mittelmäßigen F1 – die schiere Menge an Rock-Songs bedeutet auch mehr interne Variation. R&B und Jazz bilden das Schlusslicht, was auf große Überlappungen mit anderen Genres in den Feature-Räumen hindeutet. Insgesamt bestätigt das Ergebnis, dass Genre kein rein akustisches Konzept ist – Kontext, Lyrics und kulturelle Faktoren spielen eine ebenso große Rolle.

---

## Fazit

| Aufgabe | Bestes Modell | Kernergebnis |
|---|---|---|
| Popularität vorhersagen | SGD Regressor | R² = 0,17 – Künstler-Popularität schlägt Audio-Features |
| Genre klassifizieren | LinearSVC | 48,5 % Accuracy – Electronic am besten trennbar |

Der zentrale Befund dieses Projekts: **Audio-Features allein erklären weder Popularität noch Genre vollständig.** Der Bekanntheitsgrad des Künstlers ist für die Popularitätsvorhersage entscheidender als der Sound des Songs. Bei der Genre-Klassifikation zeigt sich, dass viele Genres im Audio-Raum stark überlappen – besonders Rock, Folk und Country. Für deutlich bessere Modellgüte wären Lyrics-Embeddings, Playlist-Kontext oder Ensemble-Methoden (Random Forest, Gradient Boosting) der nächste sinnvolle Schritt.
