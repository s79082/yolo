# Detektion von Rohrprofil-Stirnflächen mit YOLO auf rein synthetischen Trainingsdaten

**Technische Dokumentation — Zusammenfassung aller Arbeitsschritte, Messwerte und Ergebnisse**

Alle Zahlen in diesem Dokument sind gemessene Werte aus den Skripten des
Projekts, nicht geschätzt. Die erzeugenden Skripte sind jeweils benannt, die
Rohdaten liegen in `docs/experiments/*.json` und
`synth/*/generation_report.json`.

---

## Inhalt

1. [Aufgabenstellung und Vorgehen](#1-aufgabenstellung-und-vorgehen)
2. [Datengrundlage — Sichtung und Bewertung](#2-datengrundlage--sichtung-und-bewertung)
3. [Methode — die Augmentierungs-Pipeline](#3-methode--die-augmentierungs-pipeline)
4. [Training](#4-training)
5. [Evaluation — Test-Framework und Ergebnisse](#5-evaluation--test-framework-und-ergebnisse)
6. [Experimente](#6-experimente)
7. [Diskussion](#7-diskussion)
8. [Reproduzierbarkeit](#8-reproduzierbarkeit)

---

# 1 Aufgabenstellung und Vorgehen

## 1.1 Aufgabe

Zu detektieren sind **Stirnflächen von Rundrohrprofilen** an gestapelten
mobilen Bauzaunfeldern: der Kreisring, den man am Ende eines Rahmenrohres
sieht — heller verzinkter Kranz, dunkle Bohrung.

Zur Verfügung stehen 9 Objekt-Cutouts und 7 Hintergrundbilder. Es existieren
**keine annotierten Trainingsbilder**. Das Trainingsmaterial muss vollständig
synthetisch erzeugt werden. Validierung und Test erfolgen auf je einem echten,
handannotierten Bild.

## 1.2 Annotationsdefinition

Aus der Sichtung der annotierten Realbilder folgt eine Definition, die die
gesamte Pipeline prägt:

> Annotiert ist ausschließlich die **sichtbare** Stirnfläche. Rohrschäfte in
> Seitenansicht, Gitterknoten und dunkle Löcher im Maschendraht sind **keine**
> Objekte, obwohl sie ähnlich aussehen. Die Boxen umschließen den sichtbaren
> Teil (modale Box), nicht den geschätzten vollständigen Kreis.

## 1.3 Verarbeitungskette

```
Cutouts (9)  ──▶ [1] Matting ──────────┐
                                       ├──▶ [3] Bildsynthese ──▶ 6 000 Bilder
Hintergründe (7) ─▶ [2] Okkluder ──────┘                        136 384 Labels
                                                                      │
                                       [4] YOLO11s-Training ◀──────────┘
                                                │
                        [5] Test-Framework ◀────┴────▶ [6] Experimente
```

| Stufe | Skript | Zeilen | Ergebnis |
|---|---|---|---|
| 0 Sichtung | `src/analyze_data.py` | 196 | alle Messwerte in Kapitel 2 |
| 1 Matting | `src/matting.py` | 254 | 9 weiche Alphamasken + `ellipses.json` |
| 2 Okkluder | `src/occluders.py` | 137 | 56 echte Rohrstangen |
| 3 Bildsynthese | `src/generate.py` | 909 | 6 000 Bilder, 136 384 Labels |
| 4 Training | `src/train.py` | 281 | `runs/baseline/weights/best.pt` |
| 5 Evaluation | `src/evaluate.py` | 339 | COCO-Metriken, Zählfehler |
| 6 Experimente | `src/experiment.py` | 322 | Experimente 1, 2 und 3 |
| 7 Aufbereitung | `src/report.py` | 359 | Diagramme, Tabellen |
| — Anwendung | `src/predict.py` | 122 | Vorhersage auf eigenen Bildern |

---

# 2 Datengrundlage — Sichtung und Bewertung

Reproduzierbar über `python src/analyze_data.py`.

## 2.1 Cutouts (`/cutouts`, 9 Dateien)

| Datei | Größe (px) | Bohrung | Ovalität ry/rx | Objektanteil der Kachel | Bewertung |
|---|---|---|---|---|---|
| `cutted1.png`  | 38×32 | dunkel | 0.86 | 0.68 | brauchbar; Nachbarrohr unten rechts im Ausschnitt |
| `cutted_4.png` | 30×30 | hell   | 1.14 | 0.65 | deutlich oval, Kranz einseitig hell |
| `cutted_5.png` | 43×38 | dunkel | 0.94 | 0.73 | bestes Exemplar, größte Auflösung |
| `cutted_6.png` | 29×30 | hell   | 1.14 | 0.68 | kontrastarm |
| `cutted_7.png` | 41×35 | dunkel | 0.87 | 0.74 | sehr sauber, kräftiger Kranz |
| `cutted_8.png` | 39×35 | dunkel | 0.84 | 0.87 | **schwächstes Exemplar**: Kranz kontrastarm |
| `cutted_9.png` | 28×30 | hell   | 0.97 | 0.74 | brauchbar |
| `cuuted2.png`  | 36×33 | dunkel | 0.91 | 0.68 | bräunlicher Farbstich |
| `cuuted3.png`  | 36×34 | dunkel | 0.92 | 0.72 | brauchbar |

### Befund 1 — Der Alphakanal ist unbrauchbar (kritisch)

Alle neun Dateien sind zwar RGBA, aber der Alphakanal ist **durchgängig 255**
(gemessen: `alpha.min() == alpha.max() == 255` für alle 9). Es handelt sich um
rechteckige Bildausschnitte, nicht um freigestellte Objekte. Zwischen **13 %
und 35 %** der Kachelfläche gehören nicht zum Objekt.

**Konsequenz:** Direktes Einkopieren erzeugt an jeder Einfügestelle eine
achsparallele Naht. Ein CNN lernt diese Naht als Abkürzung („shortcut
learning") statt der Objektform — hervorragende Metriken auf synthetischen
Daten, Totalausfall auf echten. Deshalb ist das Matting (Kapitel 3.2) eine
eigene, zwingend erforderliche Pipelinestufe.

### Befund 2 — Zwei Erscheinungsformen

Sechs Cutouts haben eine **dunkle Bohrung** (offenes Rohr, Blick ins Innere),
drei eine **helle Bohrung** (Licht fällt in die Röhre, oder das Ende ist
gestaucht bzw. verschlossen). Beide Formen kommen in den Realbildern vor. Die
Pipeline zieht gleichverteilt aus allen neun; eine Gewichtung wäre ohne
Häufigkeitszählung in den Realbildern nicht belegbar.

### Befund 3 — Ovalität ist bereits im Quellmaterial vorhanden

Die aus dem Ellipsen-Fit gewonnenen Achsverhältnisse streuen zwischen 0.84 und
1.14. Ursachen sind Schrägansicht (Kreis → Ellipse in der Projektion) und
tatsächlich gequetschte Rohre. Das rechtfertigt die Ovalisierungs-Transformation
der Pipeline (T2a).

### Gesamtbewertung Cutouts

**Ausreichend, aber knapp.** Neun Instanzen sind sehr wenig; jede erscheint im
erzeugten Trainingssatz rund 15 000-mal. Das Risiko, dass das Netz die neun
konkreten Texturmuster auswendig lernt, ist real. Gegenmaßnahmen: starke
photometrische Streuung pro Instanz, Rotation über den vollen Kreis,
Ovalisierung, Dellen, Perspektive — und bewusst **kein** großes Modell
(Kapitel 4.2).

## 2.2 Hintergründe (`/bg`, 7 Dateien)

| Datei | Größe (px) | Typ | Rohr‑Ø | Mittel | σ | Schärfe (varLap) | Rauschen | Ziehgewicht |
|---|---|---|---|---|---|---|---|---|
| `test_cutted.png`  | 1513×959  | **Clean Plate** | 16 px | 70.4 | 45.1 | 1205 | 7.15 | 6 |
| `test2_cutted.jpg` | 1706×1108 | **Clean Plate** | 18 px | 97.2 | 57.8 | 1998 | 10.06 | 6 |
| `test4_cutted.png` | 1517×2788 | **Clean Plate** | 21 px | 101.9 | 60.8 | 1262 | 6.84 | 6 |
| `test5_cutted.jpg` | 945×552   | **Clean Plate** | 20 px | 102.0 | 54.0 | 2316 | 12.60 | 6 |
| `bg_3.jpg` | 1024×576 | Fremdszene | – | 159.3 | 59.7 | 1273 | 9.23 | 1 |
| `bg_4.jpg` | 800×600  | Fremdszene | – | 158.2 | 61.0 | 1779 | 10.91 | 1 |
| `bg_5.jpg` | 800×600  | Fremdszene | – | 142.4 | 40.5 | 3370 | 15.24 | 1 |

### Befund 4 — Vier echte Clean Plates (der wertvollste Teil des Materials)

`test*_cutted.*` sind Aufnahmen derselben Objektart, aus denen die Stirnflächen
herausgeschnitten wurden: Wände aus Rohrschäften **ohne** Stirnflächen. Sie
liefern drei Dinge:

1. den photometrisch exakt passenden Kontext für das Einkopieren,
2. die **Okkluder-Bibliothek** (Kapitel 3.3),
3. echte Negativbeispiele.

Sie erhalten deshalb 6-faches Ziehgewicht.

### Befund 5 — Die drei Fremdszenen sind hartes Negativmaterial

`bg_5.jpg` enthält einen Maschendrahtzaun mit rautenförmigen Öffnungen — genau
die Struktur, die als Falschpositiv droht. `bg_3.jpg` liefert Schotter (viele
dunkle, runde Kiesel). Diese Bilder sind primär als **Negativbilder** wertvoll,
nicht als Objektträger.

### Befund 6 — Der Rohrdurchmesser verankert den Maßstab

Aus den Clean Plates lässt sich der Rohrdurchmesser messen (16/18/21/20 px).
Eine Stirnfläche hat zwangsläufig **denselben** Durchmesser wie der Schaft
daneben. Der Generator koppelt deshalb Zoomfaktor und Objektgröße aneinander,
statt beide unabhängig zu ziehen (T1).

### Gesamtbewertung Hintergründe

**Gut.** Die vier Clean Plates sind qualitativ genau das Richtige. Schwäche:
alle vier zeigen dieselbe Stapelart bei bedecktem bis diffusem Licht — kein
harter Sonnenschein, keine Nässe, kein Schnee, kein Nachtbild. Diese Lücke
schließt die Pipeline nur teilweise über photometrische Augmentierung; sie
bleibt die **größte verbleibende Domänenlücke**.

## 2.3 Validierungs- und Testdaten

|  | Bild | Auflösung | Objekte |
|---|---|---|---|
| `val`  | `PXL_20251204_073853552.jpg` | 2459×1174 | 88 |
| `test` | `test5.jpg`                  | 1600×900  | 86 |

### Objektstatistik

| Größe | val | test |
|---|---|---|
| Breite (px) | 19.9 – 38.5, Median 29.3 | 9.3 – 33.3, Median 23.8 |
| Höhe (px) | 15.7 – 34.2, Median 27.0 | 6.4 – 30.5, Median 19.7 |
| √Fläche (px) | 20.6 – 35.8, **Median 27.4** | 12.2 – 31.9, **Median 19.6** |
| Seitenverhältnis w/h | p10 0.97, Median 1.08, p90 1.31 | p10 0.85, Median 1.09, p90 1.22 |
| Objektfläche gesamt | 2.48 % des Bildes | 2.69 % |
| am Rand angeschnitten | 0 % | 5.8 % |

### Anordnungsstatistik (n = 174)

| Größe | val | test |
|---|---|---|
| Abstand zum nächsten Nachbarn (in Ø) | p10 0.86, Median 1.09, p90 1.78 | p10 0.83, Median 1.24, p90 2.51 |
| Nachbarn innerhalb 5 Ø | Median 10, max 14 | Median 10, max 20 |
| Objekte, die ihren nächsten Nachbarn berühren (< 1.2 Ø) | 61.4 % | 46.5 % |
| Größenverhältnis zum nächsten Nachbarn | Median 1.05, p90 1.14 | Median 1.05, p90 1.17 |
| Nachbarrichtung (0° = horizontal) | quasi isotrop, leichte Häufung 90–150° | quasi isotrop |

### Befund 7 — Sehr kleine Objekte, dicht gepackt

Bei 20–36 px Kantenlänge in Bildern von 1600–2459 px Breite sind das
Kleinobjekte im COCO-Sinne. Die Konsequenz für die Inferenz ist zwingend:

| imgsz | Objektgröße im Netzeingang (val) | (test) |
|---|---|---|
| 640 | 7.1 px | 7.9 px |
| 960 | 10.7 px | 11.8 px |
| 1280 | 14.3 px | 15.7 px |
| 1600 | 17.8 px | 19.6 px |
| 2464 | 27.5 px | 30.2 px |

Bei `imgsz = 640` liegen die Objekte teilweise unter der Auflösung des feinsten
Detektionskopfes (P3, Stride 8). Das wird in Experiment 1 quantifiziert.

### Befund 8 — Nachbarn haben (fast) dieselbe Größe

Das Größenverhältnis benachbarter Objekte liegt im Median bei 1.05. Innerhalb
eines Stapels haben alle Rohre dieselbe Nennweite. Der Generator zieht deshalb
**eine Basisgröße pro Cluster** und variiert je Instanz nur um ±8 %, statt jede
Instanz unabhängig zu würfeln — das wäre ein sofort erkennbarer Realismusfehler.

### Befund 9 — Jede Stirnfläche hat einen Schaft

In den Nahaufnahmen gilt ausnahmslos: hinter jeder annotierten Stirnfläche
führt ein sichtbarer Rohrschaft weg. Ein dunkler Kreis mitten auf einem
durchgehenden Rohr ist physikalisch unmöglich. Diese Randbedingung ist die
wichtigste Einzelentscheidung der Pipeline (T5).

### Befund 10 — Die Datenlage der Evaluation ist die größte methodische Schwäche

**Je ein Bild** für Validierung und Test. Für eine belastbare Metrikangabe ist
das zu wenig: die Streuung einer mAP-Schätzung auf 88 Objekten ist groß, und
beide Bilder zeigen dieselbe Objektart bei ähnlichem Wetter. Alle berichteten
Zahlen sind als **Punktschätzung mit erheblicher Unsicherheit** zu lesen; das
Test-Framework gibt deshalb Bootstrap-Konfidenzintervalle aus (Kapitel 5.5).

---

# 3 Methode — die Augmentierungs-Pipeline

## 3.1 Leitprinzipien

**Prinzip 1 — Abkürzungen schließen.** Bei Copy-Paste-Synthese ist die
dominante Fehlerquelle nicht zu wenig Variation, sondern *systematische*
Unterschiede zwischen eingefügten und originalen Pixeln. Jede Transformation
wird deshalb entweder mit einer Messung an den Realdaten oder mit einer
konkreten Abkürzung begründet, die sie schließt.

**Prinzip 2 — Arbeitsteilung.**

| Generator (offline) | Ultralytics (zur Trainingszeit) |
|---|---|
| alles, was **pro Instanz** wirkt und die Box **label-exakt** neu berechnet: Rotation, Ovalisierung, Delle, Perspektive, Farbe, Okklusion | alles, was **bildweit** wirkt und Boxen nicht verfälscht: Mosaic, Skalierung, Translation, Spiegelung, HSV |

Deshalb steht im Training `degrees = 0`, `shear = 0`, `perspective = 0`: eine
Bildrotation dreht achsparallele Boxen mit und bläht sie auf (10° ⇒ +17 %
Kantenlänge bei quadratischer Box) — das wäre Labelrauschen. Im Generator wird
die Box nach der Rotation aus der gedrehten Alphamaske **exakt neu bestimmt**.

## 3.2 Stufe 1 — Matting (`src/matting.py`)

Rohrprofile sind Kreisringe; in Schrägansicht projizieren sie zu Ellipsen. Pro
Cutout wird eine Ellipse (cx, cy, rx, ry, θ) bestimmt, die die **Außenkante**
des hellen Kranzes beschreibt. Dreistufig:

1. **Manuelle Startwerte**, abgelesen aus einer 14-fach vergrößerten
   Rasteransicht. Bei 9 Objekten exakter als jeder automatische Fit.
2. **Numerische Verfeinerung** — Coordinate-Descent auf einem 6-fach
   upgesampelten, geglätteten Bild; maximiert die radiale Kantenenergie
   `I(innen) − I(außen)` entlang der Ellipse. Suchraum bewusst eng (±3 px,
   ±25°), damit der Fit nicht auf einen Nachbarring wegspringt.
3. **Außenkanten-Korrektur.** Schritt 2 konvergiert konstruktionsbedingt auf
   den *Grat* des Kranzes, nicht auf dessen Außenkante. Ein mittleres
   Radialprofil (Median über 180 Winkel, robust gegen einzelne Nachbarrohre)
   bestimmt den Punkt, an dem die Helligkeit auf **25 %** der Kantenamplitude
   über Hintergrundniveau abgefallen ist.

   *Warum 25 % und nicht der klassische 50 %-Punkt (FWHM):* der Kranz ist das
   diskriminative Merkmal. Ihn anzuschneiden wäre teurer als ein ~1 px breiter
   Hintergrundsaum, den Feathering und Farbangleichung später ohnehin
   kaschieren.

   Die Aufweitung ist hart auf **[1.02, 1.10]** gedeckelt: bei acht Cutouts
   liegt der Wert bei 1.02–1.05; nur das kontrastarme `cutted_8.png` liefe ohne
   Deckel auf 1.19 und die Maske würde Hintergrund aufsaugen.

**Ergebnis:** weiche Alphamaske mit 1.1 px Übergangsbreite. Der Übergang ist
bewusst weich — ein 25-px-Objekt hat in einer echten Aufnahme nie eine
pixelharte Silhouette. Visuelle Kontrolle: `docs/matting_check.png`.

## 3.3 Stufe 2 — Okkluder-Bibliothek (`src/occluders.py`)

**Warum echte Stangen statt gezeichneter Balken:** ein synthetisch gemalter
Balken hätte andere Textur-, Rausch- und Glanzstatistik als der Rest des
Bildes; das Netz könnte „Objekt liegt neben künstlichem Balken" lernen. Die
vier Clean Plates sind buchstäblich Wände aus Rohrschäften desselben Materials,
derselben Kamera, derselben Beleuchtung.

Verfahren:

1. Dominante Rohrrichtung über den **Struktur-Tensor**, Winkel auf (−45°, 45°]
   gefaltet, Bild entsprechend gedreht (gemessen: +0.9°, −1.6°, +0.9°, +6.6°).
2. Zeilenmittelprofil, Hochpass, lokale Maxima = **Rohrachsen**; ihr Abstand
   ist der Durchmesser (gemessen: 16, 18, 21, 20 px).
3. Streifen der Höhe = Achsabstand um jede Achse ausschneiden.
4. Weiche **Kapselmaske** (Rechteck mit halbrunden Enden, 1.6 px Feder).

**Ergebnis:** 56 Stangen mit echter zylindrischer Schattierung und echtem
Glanzlicht. Kontrolle: `docs/bars_check.png`.

## 3.4 Stufe 3 — Bildsynthese (`src/generate.py`)

Leinwand **1024 × 1024 px**, identisch zur Trainings-`imgsz` — die synthetischen
Bilder durchlaufen im Training **keine** Skalierung, die kalibrierte
Objektgrößenverteilung bleibt exakt erhalten.

Alle folgenden Anteile sind **gemessene Ist-Werte** aus
`synth/train/generation_report.json` (6 000 Bilder, 173 100 gerenderte
Instanzen, 136 384 gelabelte Objekte), nicht nur konfigurierte Sollwerte.

### T1 — Hintergrund

| Transformation | Anteil | Begründung |
|---|---|---|
| Clean Plate statt Fremdszene | 80 % (6-faches Ziehgewicht) | Die vier Clean Plates sind exakt die Zieldomäne. Die drei Fremdszenen liefern zu 20 % Kontextvielfalt und harte Negative. |
| Zoom **gekoppelt an den Rohrdurchmesser** | 100 % | **Physikalische Randbedingung.** Eine Stirnfläche hat zwangsläufig denselben Durchmesser wie der Schaft daneben, deshalb `d = pipe_d(bg) × zoom` statt zweier unabhängiger Ziehungen. Ein Missverhältnis wäre physikalisch unmöglich und würde eine falsche Kontext-Größen-Relation antrainieren. |
| Zufallsausschnitt (= **Translation**) | 100 % | Verhindert, dass immer derselbe Bildbereich als Hintergrund dient. |
| Reflect-Padding statt Kachelung | nur bei Bedarf, max. 1.35 × kürzeste Kante | `np.tile` erzeugt harte achsparallele Nähte — genau die Art Artefakt, die ein CNN als Merkmal missbraucht. Gespiegeltes Padding hat keine Kante. |
| Horizontale Spiegelung | **50.5 %** | Label-exakt und physikalisch korrekt: ein Stapel wird von beiden Enden fotografiert. |
| Vertikale Spiegelung | **14.9 %** | Ebenfalls label-exakt. Bewusst seltener als horizontal, weil die Schwerkraft (Stapel unten, Himmel oben) eine schwache Vorzugsrichtung erzeugt. |
| Kleinrotation ±6° | **34.3 %** | Die Rohrlinien sind nie exakt achsparallel. Ohne diese Variation wäre die Vorzugsrichtung im Datensatz künstlich diskret. |

### T2 — Instanz-Geometrie

| Transformation | Anteil | Begründung |
|---|---|---|
| **Rotation 0–360°** | **100 %** | Rohrenden haben keine kanonische Bildorientierung: Kamerarollwinkel und Stapelausrichtung sind beliebig. Die einzige Orientierungsinformation ist die Richtung des Glanzlichts auf dem Kranz, und die hängt vom Sonnenstand ab. Volle Rotationsinvarianz ist korrekt **und kostenlos** — es gibt keine Rotation, die ein unmögliches Bild erzeugt. Deshalb 100 % statt eines Bruchteils. |
| **Skalierung** auf 12–44 px | **100 %** | Reale Messung: val 20.6–35.8 px (Median 27.4), test 12.2–31.9 px (Median 19.6). Der Generator deckt 12–44 px ab, also die reale Spanne plus ~25 % Reserve nach oben für Aufnahmen aus geringerem Abstand. Ist-Verteilung: p05 11.6 / Median 22.5 / p95 39.9 px. Verkleinerung mit `INTER_AREA`, weil `INTER_LINEAR` Aliasing als verräterisches Hochfrequenzmuster im Objekt zurückließe. |
| **Ovalisierung** | **57.1 %** (45 % oblat 0.66–0.97, 12 % prolat 1.03–1.30) | Reale Messung: Seitenverhältnis w/h val p10–p90 = 0.97–1.31, test 0.85–1.22 — über die Hälfte der echten Boxen weicht um ≥ 5 % vom Quadrat ab. Zwei physikalische Ursachen: Schrägansicht projiziert den Kreis auf eine Ellipse, und Rohre sind tatsächlich gequetscht. Der Anteil bildet die gemessene Häufigkeit direkt ab. Oblat dominiert (45 : 12), weil Schrägansicht die Höhe verkürzt und häufiger ist als eine seitliche Quetschung. |
| **Delle** (radiale Sektor-Einbeulung) | **18.1 %** | Modelliert mechanisch beschädigte Rohrenden. Umgesetzt als glatte radiale Verschiebung über 1–2 Winkelsektoren (Gauß-Fenster, Eindringtiefe 10–30 %, quadratisch mit dem Radius gewichtet, damit das Zentrum ruhig bleibt). **Der Anteil ist eine Annahme, keine Messung** — in val/test sind Dellen nicht separat annotiert, ihre Häufigkeit ist unbekannt. 18 % wurde gewählt, damit das Modell die Verformung regelmäßig sieht (etwa jedes 5. Objekt, im Mittel 4 pro Bild), sie aber die Formstatistik nicht dominiert. Ein zu hoher Anteil würde die Rundheit als Merkmal verwässern und die Falschpositivrate auf beliebigen dunklen Flecken erhöhen. |
| **Perspektive** (Homographie ±9 %) | **30.2 %** | Bei Schrägblick wird der Kreisring nicht nur zur Ellipse — der helle Kranz wird auf einer Seite auch **breiter**, weil man die Rohrinnenwand sieht. Eine anisotrope Skalierung allein kann das nicht abbilden. 30 %, weil in val/test die Mehrheit der Stirnflächen annähernd frontal steht und ausgeprägte Schrägansicht die Ausnahme ist. |

### T3 — Instanz-Photometrie und Einbettungsrealismus

| Transformation | Anteil | Begründung |
|---|---|---|
| Helligkeit ×0.70–1.35 | **99.8 %** | Stirnflächen liegen je nach Stapelposition in praller Sonne oder tiefem Schatten. Die Spannweite ist aus den Cutouts ablesbar: mittlere Kranzhelligkeit 88–156 Grauwerte. Praktisch 100 %, weil es keine „richtige" Helligkeit gibt, die bevorzugt werden dürfte. |
| **Gamma** 0.75–1.35 | **79.7 %** | Gamma statt reinem Kontrast, weil es die nichtlineare Tonwertkurve verschiedener Kameras und JPEG-Pipelines abbildet. Es verändert vor allem das Verhältnis **dunkle Bohrung ↔ heller Kranz** — genau das Merkmal, auf das der Detektor sonst zu starr konditioniert würde. |
| HSV (Farbton ±8°, Sättigung ×0.45–1.60) | **69.7 %** | Verzinkter Stahl ist nahezu neutralgrau, nimmt aber Farbstiche an: Flugrost braun, Moos grün, blauer Himmel als Reflex. Farbtonhub bewusst klein — große Sprünge wären unphysikalisch. |
| **Farbangleich an den lokalen Hintergrund** | **84.8 %** | **Die wichtigste Anti-Abkürzungs-Maßnahme.** Cutout und Hintergrund stammen aus verschiedenen Aufnahmen mit verschiedenem Weißabgleich und Belichtungsniveau; ohne Angleich bleibt ein konstanter Farb-/Helligkeitsversatz, den ein CNN trivial als Einfügemarke lernt. Bewusst **nur teilweise**: Mittelwertangleich α ≤ 0.45, Streuungsangleich ≤ 0.35, Streuungsverhältnis auf [0.6, 1.6] geklemmt. Vollständiger Angleich würde den intrinsischen Albedo-Unterschied Bohrung/Kranz einebnen und das Objekt zerstören. Modelliert wird die *Beleuchtungs- und Weißabgleichdifferenz*, nicht die Objektfarbe. |
| **Unschärfeangleich** (σ 0.25–1.10 px) | **59.8 %** | Zwei Gründe: das skalierte Cutout ist je nach Skalierungsrichtung glatter oder schärfer als das Zielbild, und reale Objekte liegen je nach Stapeltiefe außerhalb der Schärfeebene. Schärfe ist eine der beiden Sensorsignaturen, an denen eingeklebte Pixel am leichtesten erkennbar sind. |
| **Rauschangleich** (σ aus lokaler Hochpass-Streuung geschätzt, ×0.5–1.25) | **69.9 %** | Die zweite Sensorsignatur. Beim Verkleinern eines Cutouts sinkt dessen Rauschen; der eingeklebte Bereich wäre dann messbar „sauberer" als seine Umgebung. Das Rauschniveau wird **pro Einfügestelle gemessen**, nicht global gesetzt (Hintergründe: 6.8–15.2 Grauwerte Hochpass-σ). |
| Alphakante zusätzlich weichzeichnen (σ 0.4–1.0 px) | **100 %** | Immer. Eine pixelharte Silhouette gibt es bei 25-px-Objekten in echten Aufnahmen nicht. |
| **Schlagschatten** | **49.3 %** | Ohne Kontaktschatten „schwebt" ein eingeklebtes Objekt sichtbar — es ist derselbe Hinweis, den auch ein menschlicher Betrachter zum Erkennen einer Montage nutzt. ~50 %, weil bei diffusem Licht (alle vier Clean Plates sind bedeckt aufgenommen) etwa die Hälfte der realen Objekte keinen sichtbaren Schlagschatten wirft. |

### T4 — Layout (datengetrieben)

Modell: **gescherte, verrauschte Gitterpackung** je Cluster; 1–3 Cluster pro
Bild (45 / 35 / 20 %), dazu 0–3 vereinzelte Profile.

Kalibriert an der Realmessung:

| Kenngröße | real val | real test | **synthetisch (Ist)** |
|---|---|---|---|
| NN-Abstand p10 / Median / p90 (in Ø) | 0.86 / 1.09 / 1.78 | 0.83 / 1.24 / 2.51 | **0.84 / 1.17 / 1.87** |
| Objekte, die ihren Nachbarn berühren (< 1.2 Ø) | 61.4 % | 46.5 % | **53.6 %** |
| Nachbarn innerhalb 5 Ø (Median) | 10 | 10 | **13** |
| Größenverhältnis zum Nachbarn (Median) | 1.05 | 1.05 | **1.05** (per Konstruktion) |

Zwei Modellentscheidungen folgen direkt aus der Messung:

* **Eine Basisgröße pro Cluster**, je Instanz nur ±8 % Streuung (Befund 8).
* **Cluster-Zentrum darf außerhalb der Leinwand liegen** → angeschnittene
  Objekte am Bildrand (in test 5.8 % der Labels).

Zusätzlich: **abgedunkelte Mulde** unter dem Cluster in **87.4 %** der Bilder.
In den echten Aufnahmen liegen die Stirnflächen am Stapelrand vor dem
verschatteten Stapelinneren; der Hintergrund hinter ihnen ist deutlich dunkler
als die besonnten Schäfte davor.

### T5 — Rohrschaft hinter jedem Profil (physikalische Randbedingung)

| Transformation | Anteil | Begründung |
|---|---|---|
| **Echter Rohrschaft hinter der Stirnfläche** | **88.0 %** der Instanzen | **Die wichtigste Einzelentscheidung der Pipeline** (Befund 9). Die erste Generatorversion erzeugte dunkle Kreise mitten auf durchgehenden Rohren und sah entsprechend falsch aus. Der Schaft hat exakt den Durchmesser der Stirnfläche (dasselbe Rohr, ±5 %), Länge 2–24 Ø. Nebeneffekt: Schäfte erzeugen genau die gegenseitige Verdeckung, die in den echten Bildern dominiert. |
| Schaftrichtung je Cluster parallel (±11°) | 100 % | Alle Rohre eines Stapels liegen parallel und verkanten nur leicht. ±18° erzeugte sichtbar einen sternförmigen Fächer, den es real nicht gibt. |
| Schaft-Helligkeitsangleich an die Umgebung | 100 % (k = 0.35–0.75) | Die Stangenbibliothek stammt aus vier Aufnahmen mit stark unterschiedlichem Belichtungsniveau (Hintergrundmittel 70–102 Grauwerte). Ohne Angleich klebten sichtbar zu dunkle Rohre im Bild. |
| **75 % der Schäfte ganz hinten, 25 % eingereiht** | — | **Tiefenmodell.** Die Stirnflächen bilden eine Front; die Schäfte laufen von dieser Front weg, liegen in der Projektion also *hinter* allen Stirnflächen. Nur quer vor dem Stapel liegende Rohre kreuzen vor anderen Enden. **Messbarer Effekt:** mit rein zufälliger Tiefenordnung verdeckten die Schäfte so viele Nachbarn, dass deren Labels wegfielen und der synthetische NN-Abstand auf 1.55 Ø stieg — 40 % über dem realen Wert. Mit dem 75/25-Modell: 1.17 Ø. |

### T6 — Okklusion

Okklusion entsteht aus **vier** verschiedenen, jeweils physikalisch motivierten
Quellen:

| Quelle | Anteil | Begründung |
|---|---|---|
| **Andere Profile** (Überlappung durch Tiefenordnung) | intrinsisch | Bei einem NN-Abstand von 0.84 Ø im 10. Perzentil überlappen benachbarte Stirnflächen zwangsläufig. Die Tiefenordnung ist zufällig, das Verdeckungsverhältnis damit unverzerrt. |
| **Echte Rohrstangen quer davor** | **58.4 %** der Bilder, 1–6 Stangen | Rohre, die vor dem Stapel liegen. Breite 0.55–1.35 Ø, Länge 4–26 × Breite, bevorzugt (80 %) durch ein Cluster gelegt statt ins Leere. |
| **Eigene Schäfte** (die eingereihten 25 %) | siehe T5 | Physikalisch der häufigste Verdeckungsfall in den Realbildern. |
| **Fremdkörper** (Laub, Draht, Schnee) | **31.6 %** der Bilder, 1–4 Stück | Aus den Hintergründen selbst geschnitten, mit unregelmäßiger weicher Maske aus geglättetem Rauschen. In `test4_cutted.png` liegt reichlich Laub auf dem Stapel. |
| **Bildrand** | ~3 % der Objekte | siehe T4. |

**Ergebnis:** **30.1 %** aller gelabelten Objekte sind teilverdeckt
(Sichtbarkeit < 95 %); mittlere Sichtbarkeit der gelabelten Objekte **90.0 %**.

### T7 — Sichtbarkeitsrechnung und Labelerzeugung

Kein Copy-Paste-Generator darf Labels blind vergeben. Für **jede** Instanz wird
die exakte Sichtbarkeit gerechnet:

> sichtbar_i = alpha_i · Π (1 − alpha_j)  für alle j, die **nach** i gemalt wurden

(inklusive Stangen und Fremdkörpern; die Buchhaltung läuft über die
Alpha-Fenster in Malreihenfolge).

| Regel | betroffen | Begründung |
|---|---|---|
| **Label verwerfen** wenn Sichtbarkeit < 30 % | 30 801 Objekte (17.8 % der gerenderten) | Unter 30 % sichtbarer Fläche wäre das Objekt auch für einen menschlichen Annotator nicht mehr sicher als Rohrende identifizierbar. Ein Label hier zwingt das Netz zum Halluzinieren; kein Label macht es zum korrekten Hard Negative — dieselbe Regel, nach der val/test annotiert wurden. |
| **Label verwerfen** wenn < 15 % der Fläche im Bild liegt | 3 846 Objekte | Ein Splitter am Bildrand wird real nicht annotiert. 15 % und nicht mehr, weil `test` sehr wohl stark angeschnittene Objekte enthält (kleinste Box: 0.7 % Bildhöhe). |
| **Label verwerfen** wenn sichtbare Box < 5 px | 228 Objekte | Unter der Auflösungsgrenze. |
| **Modale Box** (sichtbarer Umriss), nicht amodale | — | Genau so wurden val und test annotiert. Amodale Boxen würden einen systematischen Größenversatz gegenüber der Ground Truth erzeugen. |

### T8 — Globale Bildstufe

| Transformation | Anteil | Begründung |
|---|---|---|
| Helligkeit / Kontrast | **89.6 %** | Wetter- und Belichtungsvariation. |
| HSV / Farbtemperatur | **69.0 %** | Bewölkt (blaustichig) bis Abendsonne (warm). Alle vier Clean Plates sind bei diffusem Licht aufgenommen — diese Stufe ist der einzige Ersatz für die fehlende Beleuchtungsvielfalt. |
| Defokus (σ 0.3–1.0 px) | **34.4 %** | Autofokus trifft bei Stapelaufnahmen oft die falsche Ebene. |
| **Bewegungsunschärfe** | **9.9 %** | Verwacklung aus der Hand — plausibel, weil Baustellenaufnahmen im Vorbeigehen entstehen. Niedriger Anteil, weil moderne Handys kurz belichten. |
| Sensorrauschen | **80.1 %** | Vereinheitlicht das Rauschniveau über das ganze Bild. |
| Vignettierung | **25.0 %** | Weitwinkelobjektive von Mobiltelefonen; erzwingt Positionsinvarianz der Helligkeit. |
| **JPEG-Requantisierung q 55–95** | 85 % | **Die wichtigste „unsichtbare" Transformation.** Eingeklebte Pixel behalten sonst eine andere DCT-Blockstatistik als der Hintergrund, der bereits einmal JPEG-komprimiert war. Ein erneuter Kompressionsdurchlauf über das **Gesamtbild** vereinheitlicht diese Signatur. Das ist forensisch die belastbarste Einfügemarke überhaupt. |

### T9 — Negativbilder

**9.7 %** der Bilder enthalten **kein** Objekt: Rohrstapel ohne Stirnflächen,
Kieswerk, Betondeck mit Maschendrahtzaun.

Begründung: die dominante Falschpositiv-Quelle in dieser Domäne sind
Rohrschäfte, Gitterknoten und dunkle Löcher im Maschendraht — alles Strukturen,
die den Stirnflächen ähneln. Explizite Negativbilder bestrafen genau diese
direkt. Der Wert ~10 % entspricht der Ultralytics-Empfehlung (0–10 %
Hintergrundbilder); mehr würde den Anteil positiver Gradienten unnötig
verdünnen.

## 3.5 Der erzeugte Datensatz

| | |
|---|---|
| Bilder | 6 000 (1024 × 1024) |
| davon Negativbilder | 583 (9.7 %) |
| gerenderte Instanzen | 173 100 |
| **gelabelte Objekte** | **136 384** (22.7 pro Bild) |
| verworfen (Verdeckung / Rand / Größe) | 34 875 (20.1 %) |
| mittlere Sichtbarkeit gelabelter Objekte | 90.0 % |
| teilverdeckte gelabelte Objekte | 30.1 % |
| Objektdurchmesser p05 / Median / p95 | 11.6 / 22.5 / 39.9 px |
| Seitenverhältnis p10 / Median / p90 | 0.77 / 1.00 / 1.40 |
| Umfang auf Platte | 3.6 GB |

---

# 4 Training

## 4.1 Datenaufteilung

| Split | Quelle | Zweck |
|---|---|---|
| `synth/train` | synthetisch, 6 000 Bilder | Training |
| `data/val` | **real**, 1 Bild, 88 Objekte | Modellauswahl (`best.pt`), Early Stopping |
| `data/test` | **real**, 1 Bild, 86 Objekte | **nur** Endbewertung, berührt Training und Auswahl nie |
| `data/sval` | synthetisch, 400 Bilder | misst den Sim-to-Real-Abstand |

Die Validierung erfolgt bewusst auf dem **realen** Bild: die Modellauswahl muss
auf der Zieldomäne stattfinden, sonst wählt man das Modell aus, das am besten
zu den eigenen Renderartefakten passt.

## 4.2 Parameter mit Begründung

Modell **YOLO11s**, COCO-vortrainiert. Erzeugt aus `src/train.py`; die
Konfiguration und ihre Begründung liegen dort in einer gemeinsamen Tabelle,
damit sie nicht auseinanderlaufen können.

### Modell und Daten

| Parameter | Wert | Begründung |
|---|---|---|
| `model` | `yolo11s.pt` | **Zwei Entscheidungen, unterschiedlich gut belegt.** *Größe `s`:* der Datensatz hat genau eine Klasse und nur 9 zugrunde liegende Objektinstanzen — die Formvielfalt ist klein, die nötige Modellkapazität damit ebenfalls. `n` verliert bei 12–40 px großen Objekten Rückruf, `m`/`l` würden die synthetischen Eigenheiten auswendig lernen. COCO-vortrainierte Gewichte statt Training from scratch, weil die vortrainierten Low-Level-Filter domänenunabhängig sind. *Familie `yolo11`:* zunächst konservativ gewählt und **nicht belegt** — siehe Experiment 3 (Kapitel 6.3), das den Vergleich nachholt. |
| `imgsz` | `1024` | Identisch zur Leinwandgröße des Generators. Damit durchlaufen die synthetischen Bilder **keine** Skalierung und die Objekte erreichen das Netz mit exakt der Pixelgröße (12–44 px), für die die Pipeline ausgelegt wurde. Bei 640 px wären die Objekte nur noch 7–27 px groß und lägen teils unter der Auflösung des P3-Kopfes (Stride 8). |
| `epochs` | `60` | 6 000 Bilder / Batch 16 = 375 Iterationen je Epoche. Bei einem Ein-Klassen-Problem mit vortrainiertem Backbone ist die Konvergenz typischerweise nach 30–50 Epochen erreicht; 60 gibt der Kosinus-Rampe Raum zum Ausschwingen. **Tatsächlich beendet durch Early Stopping nach 30 Epochen.** |
| `patience` | `20` | Early Stopping nach 20 Epochen ohne Verbesserung. Bewusst großzügig: die Fitness schwankt auf nur 88 Ground-Truth-Objekten stark, ein knappes Fenster würde zu früh abbrechen. |
| `batch` | `16` | Belegt rund 11.6 GB. Feste Zahl statt `batch=-1`, weil Autobatch je nach Fremdlast auf der GPU unterschiedliche Werte wählt und der Lauf dann nicht reproduzierbar ist. |
| `device` | `0` | GPU 0 (RTX PRO 6000 Blackwell). |
| `seed`, `deterministic` | `0`, `True` | Voraussetzung dafür, dass die Ablationen in Experiment 2 vergleichbar sind. |
| `amp` | `True` | Mixed Precision. Halbiert Speicher und Rechenzeit; Ultralytics hält Verlustberechnung und Normierung in FP32, der Genauigkeitsverlust ist bei Detektion nicht messbar. |
| `workers` | `0` | **Erzwungen, nicht gewählt** — siehe Kapitel 8.2. |
| `cache` | `ram` | 6 000 × 1024 × 1024 × 3 B = 18 GB bei 94 GB frei. Weil `workers=0` erzwungen ist, läuft die gesamte Datenaufbereitung im Hauptprozess; das Wegfallen der JPEG-Dekodierung ist dort der größte einzelne Zeitgewinn (Faktor ~1.6 gemessen). Der Cache wird von einem Thread-Pool gefüllt, nicht von Prozessen, und ist vom Deadlock nicht betroffen. |

### Optimierung

| Parameter | Wert | Begründung |
|---|---|---|
| `optimizer` | `AdamW` | Explizit gesetzt statt `auto`. AdamW konvergiert bei kleinen Datensätzen deutlich schneller als SGD und ist gegenüber der Lernratenwahl robust — wichtig, weil kein Budget für eine Lernraten-Suche vorhanden ist. Die entkoppelte Gewichtsdämpfung verhindert, dass der Weight Decay mit der adaptiven Schrittweite interagiert. |
| `lr0` | `0.001` | Standard-Startlernrate für AdamW-Finetuning. Höhere Werte (3e-3) zerstören die vortrainierten Backbone-Gewichte in den ersten Epochen, niedrigere (3e-4) verlangsamen ohne Gewinn. |
| `lrf` | `0.01` | Endlernrate = 1 % der Startlernrate. Die letzten Epochen dienen nur noch der Feinjustierung der Box-Regression, was gerade bei 20–30 px großen Objekten den Lokalisierungsfehler senkt. |
| `cos_lr` | `True` | Kosinus-Abfall statt linear: längeres Verweilen auf hoher Lernrate (bessere Exploration), dann weiches Auslaufen. |
| `warmup_epochs` | `3.0` | Ohne Warmup erzeugt der frisch initialisierte Detektionskopf in den ersten Schritten sehr große Gradienten, die in den vortrainierten Backbone zurückschlagen. |
| `weight_decay` | `0.0005` | Ultralytics-Standard. Bei nur 9 Quellobjekten ist eine gewisse Regularisierung erwünscht. |
| `momentum` | `0.937` | Entspricht β₁ bei AdamW (Ultralytics-Konvention). |

### Verlustgewichte

| Parameter | Wert | Begründung |
|---|---|---|
| `box` | `7.5` | Standardgewicht der CIoU/DFL-Boxverluste. Eine Erhöhung wäre bei kleinen Objekten denkbar, wurde hier aber **nicht** vorgenommen: die Zielmetrik ist überwiegend mAP@0.5 und Zählgenauigkeit, nicht Pixelpräzision, und ohne separaten Tuning-Split wäre jede Abweichung vom Standard nicht belegbar. |
| `cls` | `0.5` | Bei einer einzigen Klasse ist der Klassifikationsverlust faktisch eine Objektheits-Entscheidung. Das Verhältnis box : cls steuert die Balance zwischen Finden und Lokalisieren und kann ohne Tuning-Split nicht seriös verschoben werden. |
| `dfl` | `1.5` | Standardgewicht der Distribution Focal Loss. |

### Trainingszeit-Augmentierung

| Parameter | Wert | Begründung |
|---|---|---|
| `mosaic` | `1.0` | Immer aktiv. Setzt vier Bilder zu einem zusammen: erhöht die Objektzahl pro Iteration, bringt Objekte an Kachelränder (realistisch — in test sind 5.8 % der Labels angeschnitten) und kombiniert verschiedene Hintergründe in einem Bild. Bei dichten Kleinobjekten die wirksamste Einzelaugmentierung. |
| `close_mosaic` | `10` | Mosaic in den letzten 10 Epochen abschalten. Mosaic erzeugt eine Statistik, die es bei der Inferenz nie gibt (Vierfachkacheln); das Abschalten am Ende lässt das Modell auf der echten Bildstatistik ausklingen. |
| `scale` | `0.4` | Zufällige Skalierung 0.6–1.4. Zusammen mit der Generatorspanne 12–44 px ergibt das effektiv 7–62 px und deckt alle Inferenzauflösungen aus Experiment 1 ab. Der Ultralytics-Standard 0.5 wurde reduziert, weil er die kleinsten Objekte auf 6 px schrumpfen lässt — unterhalb dessen, was Stride 8 auflöst, und damit reines Labelrauschen. |
| `translate` | `0.1` | Standard. Verschiebt das Mosaic-Zentrum, verhindert eine Positionsverzerrung zur Bildmitte. |
| `degrees` | `0.0` | **Bewusst aus.** Eine Bildrotation dreht achsparallele Boxen mit und bläht sie auf (10° → +17 % Kantenlänge bei quadratischer Box), erzeugt also Labelrauschen. Die nötige Rotationsinvarianz liefert der Generator: dort wird **jede** Instanz um 0–360° gedreht und die Box danach aus der gedrehten Alphamaske exakt neu bestimmt. |
| `shear`, `perspective` | `0.0` | Aus, gleiche Begründung. Perspektive wird im Generator pro Instanz label-exakt angewandt (30 % der Instanzen). |
| `fliplr` | `0.5` | Label-exakt und physikalisch korrekt: ein Stapel kann von beiden Enden fotografiert werden. |
| `flipud` | `0.5` | Ebenfalls label-exakt. Ungewöhnlich für Naturbilder, hier aber richtig: die Kamera wird in beliebiger Rollage gehalten, und ein Rohrende hat kein „oben". |
| `hsv_h` | `0.015` | Sehr kleiner Farbtonversatz — verzinkter Stahl ist neutral, große Farbtonsprünge wären unphysikalisch. |
| `hsv_s` | `0.7` | Kräftige Sättigungsvariation deckt die Spanne von neutralgrau bis rostbraun / moosgrün ab. |
| `hsv_v` | `0.4` | Helligkeitsvariation für Sonne / Schatten / Bewölkung. |
| `mixup` | `0.0` | **Aus.** Mixup überlagert zwei Bilder halbtransparent und erzeugt Geisterobjekte mit vollem Label. Bei 20 px großen, dicht gepackten Objekten ist das reines Labelrauschen. |
| `cutmix` | `0.0` | **Aus.** Okklusion wird im Generator physikalisch korrekt erzeugt (echte Rohrstangen), nicht durch eingesetzte Rechtecke aus fremden Bildern. |
| `erasing` | `0.0` | **Aus.** Wirkt in Ultralytics ohnehin nur bei Klassifikation; die entsprechende Funktion übernimmt die Okklusionsstufe des Generators. |
| `copy_paste` | `0.0` | **Aus** — benötigt Segmentierungsmasken und ist für die Detektionsaufgabe nicht verfügbar. Copy-Paste ist genau das, was der Generator offline und mit korrekter Sichtbarkeitsrechnung leistet. |

## 4.3 Trainingsverlauf

![Trainingsverlauf](fig_training.png)

| | |
|---|---|
| gelaufene Epochen | 30 (Early Stopping, Obergrenze 60) |
| bestes Modell | **Epoche 10** |
| Laufzeit | 1.13 h |
| Durchsatz | 3.0 it/s bei Batch 16 ≈ 50 Bilder/s |
| GPU-Speicher | 11.6 GB |

Der Trainingsverlust fällt über alle 30 Epochen monoton, die Metrik auf dem
realen `val`-Bild ist aber ab Epoche ~10 flach. Ab dort verbessert das Modell
nur noch seine Anpassung an die synthetische Verteilung. Genau deshalb ist es
richtig, `best.pt` über den **realen** Validierungssplit auszuwählen.

**Checkpoint-Auswahl:** `best.pt` und `last.pt` wurden zusätzlich bei nativem
Maßstab auf `val` verglichen — `best.pt` gewinnt deutlich (mAP@0.5:0.95 0.597
gegen 0.565) und wird für alle Auswertungen verwendet.

---

# 5 Evaluation — Test-Framework und Ergebnisse

## 5.1 Warum ein eigenes Framework

`model.val()` von Ultralytics wurde nicht als alleinige Grundlage verwendet:

1. **Nachvollziehbarkeit** — die Metrik muss selbst implementiert und
   erklärbar sein, nicht Blackbox.
2. **Anwendungsnähe** — für das eigentliche Ziel (Rohre zählen) ist der
   Zählfehler wichtiger als mAP@0.5:0.95.
3. **Freie Variation aller Inferenzparameter** inklusive Sweeps.

Die Ultralytics-Zahlen werden als Kreuzvergleich mitgeführt (Kapitel 5.4).

## 5.2 Metrikdefinitionen

* **Zuordnung** greedy nach absteigender Konfidenz, jede Ground-Truth-Box nur
  einmal.
* **AP** über 101-Punkt-Interpolation des Precision-Recall-Verlaufs (COCO).
* **mAP@0.5:0.95** = Mittel der AP über die IoU-Schwellen 0.50, 0.55, …, 0.95.
* **Betriebspunkt** = die Konfidenzschwelle, die F1 maximiert (relevant für den
  praktischen Einsatz, wo eine feste Schwelle gewählt werden muss).
* **Zählfehler** = |vorhergesagte Anzahl − Ground-Truth-Anzahl| am
  F1-optimalen Betriebspunkt.

## 5.3 Inferenzparameter

| Parameter | Wert | Begründung |
|---|---|---|
| `imgsz` | **1920** (auf `val` bestimmt) | Der dominante Parameter dieser Aufgabe — siehe Experiment 1a. Die Objekte sind nativ 12–36 px groß; bei imgsz 640 schrumpfen sie auf 5–9 px und liegen unter der Auflösung des feinsten Detektionskopfes. |
| `conf` | 0.001 für mAP, 0.24–0.34 als Betriebspunkt (0.54 mit TTA) | Für die mAP-Berechnung 0.001, sonst wird der Recall-Ast der PR-Kurve abgeschnitten. Der Betriebspunkt wird aus der F1-Kurve bestimmt und separat berichtet. |
| `iou` (NMS) | 0.7 | Unkritisch: benachbarte Rohrenden haben mindestens ~0.85 Durchmesser Abstand, ihre Boxen überlappen kaum. Ein zu niedriger Wert würde dicht stehende, tatsächlich verschiedene Rohre gegenseitig auslöschen. |
| `max_det` | 1000 | Der Ultralytics-Standard 300 reicht für diese Bilder, wäre bei einem großen Stapel aber eine stille Obergrenze. 1000 stellt sicher, dass die Metrik nie durch eine Kappungsgrenze verfälscht wird. |
| `augment` (TTA) | optional | +2.1 Punkte mAP@0.5 auf `val`, doppelte Rechenzeit — siehe Experiment 1d. |
| `quantize` | `None` (FP32) | FP16 liefert bitidentische Ergebnisse (Experiment 1d). |
| `rect` | implizit aktiv | Padding nur auf Vielfache von 32 statt auf ein Quadrat. |
| Nachbearbeitung | keine | Keine Mindestgröße, keine Randunterdrückung, keine Cluster-Nachbearbeitung. Jede solche Regel wäre ein an `val` angepasster Freiheitsgrad und würde die Aussagekraft der `test`-Zahlen schwächen. |

### Der Betriebspunkt ist der am schlechtesten übertragbare Parameter

Für den praktischen Einsatz muss eine feste Konfidenzschwelle gewählt werden.
Zwei Kriterien bieten sich an: die **F1-optimale** Schwelle (maximiert das
harmonische Mittel aus Precision und Recall) und die **zähloptimale** Schwelle
(dort heben sich Falschpositive und Falschnegative gerade auf, die Zählung
stimmt exakt). Beide wurden auf `val` bestimmt und dann auf `test` angewandt —
das ist die ehrliche Simulation des Einsatzfalls.

| Modell | Kriterium | conf (aus val) | test F1 | test P | test R | test Zählung |
|---|---|---|---|---|---|---|
| `yolo11s` | F1-optimal | 0.243 | 0.907 | 0.907 | 0.907 | 86 / 86 (±0) |
| `yolo11s` | zähloptimal | 0.373 | 0.917 | 0.939 | 0.895 | 82 / 86 (−4) |
| `yolo26s` | F1-optimal | 0.257 | 0.867 | 0.900 | 0.837 | 80 / 86 (−6) |
| `yolo26s` | zähloptimal | 0.350 | 0.857 | 0.920 | 0.802 | 75 / 86 (−11) |
| `yolo26s-p2` | F1-optimal | 0.439 | **0.929** | **0.951** | 0.907 | 82 / 86 (−4) |
| `yolo26s-p2` | zähloptimal | 0.701 | 0.912 | 0.986 | 0.849 | 74 / 86 (−12) |

**Die zähloptimale Schwelle überträgt sich bei allen drei Modellen schlechter
als die F1-optimale.** Der Grund ist geometrisch: der zähloptimale Punkt liegt
auf einer Messerschneide — er ist gerade dort, wo sich zwei gegenläufige Fehler
zufällig aufheben, und diese Balance ist bildspezifisch. Die F1-Kurve dagegen
hat ein breites Plateau (siehe `fig_pr_curves.png`), auf dem eine Verschiebung
der Schwelle wenig kostet.

> **Empfehlung:** Betriebspunkt über das F1-Maximum wählen, nicht über die
> Zählgenauigkeit — auch dann, wenn Zählen die eigentliche Anwendung ist.

Dass `yolo11s` mit der F1-optimalen Schwelle auf `test` exakt 86 von 86 trifft,
ist Zufall: die auf `val` bestimmte Schwelle landet dort gerade auf dem
Nulldurchgang. Mit einem Bild pro Split lässt sich daraus nichts ableiten.

## 5.4 Ergebnisse

Bestes Modell, nativer Maßstab (`imgsz = native`):

| Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | conf* | P | R | mittl. IoU (TP) | Zählung |
|---|---|---|---|---|---|---|---|---|---|
| `val` (real) | 0.886 | 0.793 | 0.597 | 0.862 | 0.336 | 0.872 | 0.852 | 0.824 | 86 / 88 |
| `test` (real, unberührt) | 0.888 | 0.320 | 0.437 | 0.894 | 0.263 | 0.905 | 0.884 | 0.748 | 84 / 86 |
| `sval` (synthetisch) | 0.982 | 0.935 | 0.828 | 0.964 | 0.450 | 0.972 | 0.956 | 0.899 | 7128 / 7248 |

Bei der auf `val` gewählten Auflösung `imgsz = 1920`: `test` mAP@0.5 = **0.928**,
mit TTA **0.934**.

**Zählfehler: 2.3 %** auf beiden realen Bildern.

![PR-Kurven und F1 über Konfidenz](fig_pr_curves.png)

### Verifikation gegen Ultralytics

| | eigene Implementierung | `model.val()` |
|---|---|---|
| `val` mAP@0.5 / mAP@0.5:0.95 | 0.886 / 0.597 | 0.897 / 0.599 |
| `test` mAP@0.5 / mAP@0.5:0.95 | 0.888 / 0.437 | 0.891 / 0.424 |

Abweichung ≤ 1.3 Prozentpunkte, erklärbar durch Detailunterschiede in der
AP-Interpolation und der Zuordnungsreihenfolge. Die eigene Implementierung ist
damit verifiziert.

## 5.5 Bootstrap-Konfidenzintervalle

Weil je Split nur **ein** Bild vorliegt, ist eine Punktschätzung ohne
Streuungsmaß nicht interpretierbar. Das Framework resampelt deshalb
**räumliche Kacheln** (6×6-Raster, 400 Wiederholungen). Kacheln statt
Einzelobjekten, weil Detektionen und Ground Truth räumlich stark korreliert
sind (Cluster) — ein Einzelobjekt-Bootstrap würde die Varianz deutlich
unterschätzen.

| Split | Mittel | Std | 95 %-Konfidenzintervall | Kacheln |
|---|---|---|---|---|
| `val` | 0.892 | 0.051 | [0.766, 0.965] | 23 |
| `test` | 0.911 | 0.036 | [0.844, 0.974] | 25 |

---

# 6 Experimente

## 6.1 Experiment 1 — Inferenzparameter

Kein erneutes Training nötig; ausgewertet auf dem Baseline-Modell.

### 1a Inferenzauflösung (`imgsz`)

![imgsz-Sweep](fig_imgsz_sweep.png)

| imgsz | `val` mAP@0.5 | `val` mAP@0.5:0.95 | `test` mAP@0.5 | `test` mAP@0.5:0.95 |
|---|---|---|---|---|
| 640 | 0.435 | 0.201 | 0.524 | 0.335 |
| 800 | 0.793 | 0.428 | 0.665 | 0.363 |
| 960 | 0.852 | 0.498 | 0.889 | 0.430 |
| 1024 | 0.868 | 0.546 | 0.882 | 0.459 |
| 1280 | 0.887 | 0.577 | 0.900 | 0.449 |
| 1600 | **0.904** | 0.606 | 0.888 | 0.437 |
| 1920 | 0.893 | **0.613** | 0.928 | 0.448 |
| 2240 | 0.880 | 0.588 | **0.928** | **0.457** |
| 2464 | 0.886 | 0.597 | 0.927 | 0.454 |

**Von 640 auf 1600 px verdoppelt sich mAP@0.5 auf `val` (0.435 → 0.904).** Das
ist mit Abstand der größte Einzeleffekt der gesamten Arbeit — größer als jede
Entscheidung in der Augmentierungs-Pipeline.

Die Erklärung ist rein geometrisch und wurde in Befund 7 vorhergesagt: bei
imgsz 640 wird `val` (2459 px breit) um den Faktor 0.26 verkleinert, die
Objekte schrumpfen auf 5–9 px. Der feinste Detektionskopf von YOLO11 arbeitet
auf Stride 8 — ein 6-px-Objekt belegt dort weniger als ein Ausgabepixel.

Ab etwa 1280 px sättigt die Kurve. Bemerkenswert: **das Optimum liegt nicht am
nativen Maßstab.** Für `test` (1600 px breit) ist imgsz 1920–2464, also eine
*Hochskalierung* um Faktor 1.2–1.5, spürbar besser als der native Maßstab
(0.928 gegen 0.888). Der Grund ist die Trainingsverteilung: der Generator
erzeugt Objekte mit Median 22.5 px, `test` hat aber nur 19.6 px Median. Eine
leichte Hochskalierung schiebt die Testobjekte in die Mitte der
Trainingsverteilung.

> **Praktische Empfehlung:** nicht auf den nativen Maßstab skalieren, sondern
> so, dass die erwartete Objektgröße etwa 22 px im Netzeingang beträgt.

**Auswahlmethodik:** die beste Auflösung wurde ausschließlich auf `val`
bestimmt (1920) und dann auf beide Splits angewandt. Eine Wahl je Split wäre
eine Hyperparameterselektion auf dem Testsatz und würde dessen Zahlen wertlos
machen.

### mAP@0.75 zeigt die Grenze der Lokalisierung bei Kleinstobjekten

`val` erreicht mAP@0.75 = 0.793, `test` nur 0.320 — bei fast identischem
mAP@0.5. Der Unterschied ist die Objektgröße (Median 27.4 px gegen 19.6 px).
Bei einem 20-px-Objekt kostet ein Boxfehler von 2 px bereits ~0.2 IoU. Die
mittlere IoU der Treffer bestätigt das: 0.824 auf `val`, 0.748 auf `test`.
mAP@0.5:0.95 ist bei dieser Objektgröße folglich mehr ein Maß für die
Pixelauflösung als für die Detektionsqualität.

### 1b NMS-IoU (bei imgsz 1920)

| `iou` | `val` mAP@0.5 | `val` mAP@0.5:0.95 | `test` mAP@0.5 | `test` mAP@0.5:0.95 |
|---|---|---|---|---|
| 0.30 | **0.897** | **0.620** | 0.933 | 0.453 |
| 0.45 | 0.895 | 0.616 | 0.931 | 0.450 |
| 0.60 | 0.892 | 0.614 | 0.929 | 0.448 |
| 0.70 | 0.893 | 0.613 | 0.928 | 0.448 |
| 0.80 | 0.887 | 0.609 | **0.934** | 0.449 |
| 0.90 | 0.867 | 0.600 | 0.928 | 0.450 |

Spanne nur 3 Punkte mAP@0.5. Wie in Kapitel 5.3 vorhergesagt: benachbarte
Rohrenden überlappen kaum, NMS ist hier fast wirkungslos. Der Standard 0.7
verliert nichts Nennenswertes.

### 1c `max_det` (bei imgsz 1920)

| `max_det` | `val` mAP@0.5 | `val` Zählung | `test` mAP@0.5 | `test` Zählung |
|---|---|---|---|---|
| 50 | 0.735 | 50 / 88 | 0.788 | 50 / 86 |
| 100 | 0.885 | 100 / 88 | **0.944** | 82 / 86 |
| 200 | 0.892 | 100 / 88 | 0.932 | 82 / 86 |
| 300 | 0.892 | 100 / 88 | 0.930 | 82 / 86 |
| 1000 | 0.893 | 100 / 88 | 0.928 | 82 / 86 |

`max_det = 50` ist katastrophal — die Kappung schneidet ein Drittel der Objekte
ab. Ab 100 gesättigt.

### 1d Test-Time-Augmentation und FP16 (bei imgsz 1920)

| Variante | `val` mAP@0.5 | `val` F1 | `val` Zählung | `test` mAP@0.5 | `test` F1 | `test` Zählung | Zeit |
|---|---|---|---|---|---|---|---|
| Standard | 0.893 | 0.851 | 100 / 88 | 0.928 | 0.917 | 82 / 86 | 1× |
| **TTA** | **0.914** | **0.899** | **90 / 88** | **0.934** | **0.918** | **84 / 86** | ~2× |
| FP16 | 0.893 | 0.851 | 100 / 88 | 0.928 | 0.917 | 82 / 86 | 1× |
| TTA + FP16 | 0.914 | 0.899 | 90 / 88 | 0.934 | 0.918 | 84 / 86 | ~2× |

TTA bringt **+2.1 Punkte** mAP@0.5 auf `val`, **+0.6** auf `test`, und deutlich
bessere Zählung. Für eine Zählanwendung, die nicht in Echtzeit laufen muss,
lohnt sich das. FP16 liefert bitidentische Ergebnisse — kein Gewinn, aber auch
kein Grund, es nicht zu verwenden.

## 6.2 Experiment 2 — Okklusionsanteil in den Trainingsdaten

### Versuchsaufbau

Drei Trainingsläufe mit **identischem Budget** (3 000 Bilder, 30 Epochen,
Seed 1, identische Hyperparameter). Einziger Unterschied ist der
Okklusionsanteil in den erzeugten Daten. Der Rohrschaft selbst bleibt in allen
Armen erhalten — er ist physikalische Randbedingung, keine Augmentierung.

| Arm | Stangen | Fremdkörper | Schäfte vor der Front |
|---|---|---|---|
| **A0** | aus | aus | 0 % |
| **A1** (Baseline-Rezept) | 58 % der Bilder | 32 % | 25 % |
| **A2** | 90 % der Bilder | 64 % | 60 % |

![Dasselbe Bild aus allen drei Armen](fig_ablation_beispiele.png)

### Erzeugte Datensätze

| Arm | mittlere Sichtbarkeit | teilverdeckt | Labels | wegen Verdeckung verworfen |
|---|---|---|---|---|
| **A0** | 0.977 | 12.0 % | 82 950 | 241 |
| **A1** | 0.902 | 29.9 % | 68 281 | 15 312 |
| **A2** | 0.852 | 40.7 % | 55 833 | 28 108 |

### Ergebnis

![Ablation](fig_ablation.png)

| Arm | `val` mAP@0.5 | `val` mAP@0.5:0.95 | `test` mAP@0.5 | `test` mAP@0.5:0.95 | `test` F1 |
|---|---|---|---|---|---|
| **A0** ohne Okklusion | 0.888 | 0.606 | 0.899 | 0.418 | 0.896 |
| **A1** Baseline | 0.924 | 0.609 | 0.911 | 0.452 | 0.904 |
| **A2** starke Okklusion | **0.930** | **0.620** | **0.919** | **0.454** | **0.910** |

**Der Effekt ist auf beiden Splits, in allen drei Metriken und über alle drei
Stufen monoton.** A0 → A2 bringt +2.0 Punkte mAP@0.5 und +3.6 Punkte
mAP@0.5:0.95 auf dem unberührten Testbild.

Bemerkenswert ist, **wogegen** dieser Gewinn erzielt wird: A2 hat durch die
stärkere Verdeckung **33 % weniger Labels** als A0 bei gleicher Bildzahl. Der
Okklusionsvorteil überkompensiert also einen deutlichen Verlust an Supervision.
Das ist der stärkste Beleg dafür, dass die Okklusionsstufe kein Beiwerk ist.

Ein zweiter Befund steckt im Trainingsverlauf: **A0 erreichte sein bestes
Modell bereits in Epoche 3** und verbesserte sich in den folgenden 20 Epochen
nicht mehr (Early Stopping bei Epoche 23). A1 und A2 verbesserten sich bis zur
letzten Epoche 30 durchgehend. Ohne Okklusion ist die Aufgabe für das Netz zu
leicht — es sättigt sofort und passt sich danach nur noch an die synthetische
Verteilung an.

### Einschränkungen dieses Experiments

1. **Der Effekt liegt innerhalb des Konfidenzintervalls.** Das Bootstrap-CI für
   mAP@0.5 ist ±0.05 (`val`) bzw. ±0.04 (`test`) — breiter als der gemessene
   Unterschied von 0.02–0.04. Belastbar ist nicht die einzelne Differenz,
   sondern die **Konsistenz**: dieselbe Richtung in 3 Metriken × 2 Splits × 3
   Stufen. Bei reinem Rauschen wäre eine solche durchgehende Monotonie
   unwahrscheinlich, ein formaler Signifikanznachweis ist mit je einem Bild pro
   Split aber nicht möglich.
2. **Labelzahl und Okklusionsanteil lassen sich nicht entkoppeln.** Mehr
   Verdeckung heißt zwangsläufig mehr verworfene Labels. Ein sauberer
   Nachfolgeversuch müsste die Bildzahl je Arm so anpassen, dass alle Arme
   gleich viele Labels haben.
3. **A1 und A2 waren nach 30 Epochen noch nicht auskonvergiert.** Ihr Vorsprung
   könnte bei längerem Training größer werden.
4. A1 und A2 (3 000 Bilder, 30 Epochen) schlagen den Hauptlauf (6 000 Bilder)
   auf beiden Splits. Auch das liegt im Rauschen, deutet aber darauf hin, dass
   mehr synthetische Bilder ab einer gewissen Menge nichts mehr bringen — die
   Vielfalt ist durch die 9 Cutouts begrenzt, nicht durch die Bildzahl.

## 6.3 Experiment 3 — Modellfamilie

### Motivation

Die Wahl von YOLO11s war in der ursprünglichen Fassung konservativ begründet
(ausgereifte Familie, stabile COCO-Gewichte, bekannte Kopfstruktur), aber
**nicht gemessen**. Tatsächlich ist in der verwendeten Ultralytics-Version
8.4.117 **YOLO26 der Default** — der AMP-Check lädt `yolo26n.pt`, und
`check_suffix()` verwendet denselben Standard. YOLO12- und YOLO26-Konfigurationen
liegen der Bibliothek bei. Dieses Experiment holt den Beleg nach.

### Architektonische Unterschiede

| Modell | Parameter | Strides | `reg_max` | NMS | Trainingszeit | GPU-Speicher |
|---|---|---|---|---|---|---|
| `yolo11s` | 9.43 M | 8 / 16 / 32 | 16 | ja | 1.13 h | 11.6 GB |
| `yolo26s` | 9.95 M | 8 / 16 / 32 | 1 | **nein** (end2end) | 0.96 h | 14.5 GB |
| `yolo26s-p2` | 9.66 M | **4** / 8 / 16 / 32 | 1 | **nein** (end2end) | 2.02 h | 29.9 GB |

Drei Punkte sind dabei fachlich relevant:

1. **Alle drei haben P3 (Stride 8) als feinste Standard-Ebene.** Die gesamte
   Auflösungsargumentation aus Befund 7 und Experiment 1a gilt damit für alle
   Modelle unverändert.
2. **YOLO26 ist end-to-end, also NMS-frei.** Experiment 1b (NMS-IoU-Sweep) ist
   dort gegenstandslos. Bei dicht gepackten Objekten ist das potenziell ein
   Vorteil, weil eine Fehlerquelle ganz entfällt.
3. **`reg_max = 1` statt 16** bedeutet: keine Distribution Focal Loss über
   Boxbins. Bei 20-px-Objekten, wo die Lokalisierung ohnehin die Schwachstelle
   ist, war offen, in welche Richtung das wirkt.

`yolo26s-p2` verwendet die mitgelieferte Konfiguration `yolo26-p2.yaml` in
Größe `s`; die passenden Schichten wurden aus `yolo26s.pt` übernommen
(696 von 708 Tensoren), der zusätzliche P2-Kopf startet zufällig initialisiert.

### Versuchsaufbau

Identisches Budget für alle drei: derselbe Datensatz (`synth/train`, 6 000
Bilder), `imgsz = 1024`, `batch = 16`, `epochs = 60`, `patience = 20`,
`seed = 0`, identische Optimierungs- und Augmentierungsparameter. Ausgewertet
mit demselben Test-Framework.

### Ergebnis bei nativem Maßstab

![Modellvergleich](fig_models.png)

| Modell | Split | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | bestes F1 | P | R | mittl. IoU (TP) | Zählung |
|---|---|---|---|---|---|---|---|---|---|
| `yolo11s` | val | 0.886 | 0.793 | 0.597 | 0.862 | 0.872 | 0.852 | 0.824 | 86 / 88 |
| `yolo11s` | test | 0.888 | 0.320 | 0.437 | 0.894 | 0.905 | 0.884 | 0.748 | 84 / 86 |
| `yolo26s` | val | 0.931 | 0.795 | **0.624** | 0.891 | 0.854 | 0.932 | 0.824 | 96 / 88 |
| `yolo26s` | test | 0.905 | 0.333 | 0.427 | 0.888 | 0.904 | 0.872 | 0.729 | 83 / 86 |
| `yolo26s-p2` | val | **0.936** | 0.750 | 0.607 | 0.897 | 0.856 | **0.943** | 0.813 | 97 / 88 |
| `yolo26s-p2` | test | **0.941** | **0.387** | **0.450** | **0.933** | **0.975** | **0.895** | 0.728 | 79 / 86 |

### Ergebnis bei der je Modell auf `val` gewählten Auflösung

Die Auflösung wurde für jedes Modell separat auf `val` optimiert und dann auf
`test` angewandt — keine Selektion auf dem Testsatz.

| Modell | imgsz | val mAP@0.5 | val mAP@0.5:0.95 | test mAP@0.5 | test mAP@0.5:0.95 | test F1 | test Zählung |
|---|---|---|---|---|---|---|---|
| `yolo11s` | 1920 | 0.893 | 0.613 | 0.928 | **0.448** | 0.917 | 82 / 86 |
| `yolo26s` | 1600 | 0.939 | 0.638 | 0.905 | 0.427 | 0.888 | 83 / 86 |
| `yolo26s-p2` | 1920 | **0.940** | **0.642** | **0.947** | 0.449 | **0.934** | 81 / 86 |

### Interpretation

**Die ursprüngliche Modellwahl war suboptimal.** `yolo26s-p2` erreicht auf dem
unberührten Testbild mAP@0.5 = 0.941 (nativ) bzw. 0.947 (bei val-optimierter
Auflösung) gegenüber 0.888 bzw. 0.928 für `yolo11s` — ein Unterschied von 5.3
bzw. 1.9 Prozentpunkten. Beim F1 sind es 0.933 gegen 0.894 (nativ).

**Der P2-Kopf half — aber nicht auf die vorhergesagte Weise.** Die Hypothese im
ursprünglichen Ausblick lautete: ein Stride-4-Kopf verbessert die
Lokalisierungspräzision und damit mAP@0.75 auf `test` (0.320). Gemessen:
mAP@0.75 steigt auf 0.387 (+6.7 Punkte) — die Vorhersage trifft also zu. Die
**mittlere IoU der Treffer** verbessert sich jedoch *nicht* (0.728 gegen 0.748
bei `yolo11s`). Der mAP@0.75-Gewinn kommt daher überwiegend daher, dass
**mehr Objekte überhaupt gefunden werden** (Recall 0.895 gegen 0.884, mAP@0.5
+5.3 Punkte), nicht daraus, dass die gefundenen präziser umschlossen würden.
Die Hypothese ist damit nur zur Hälfte bestätigt.

**`reg_max = 1` kostet bei kleinen Objekten etwas.** `yolo26s` liegt auf `val`
bei mAP@0.5:0.95 vorn (0.624 gegen 0.597), auf `test` aber knapp zurück (0.427
gegen 0.437) — dort sind die Objekte kleiner (Median 19.6 px gegen 27.4 px).
Das passt zu der Erwartung, dass der Wegfall der DFL-Verteilung gerade bei
Kleinstobjekten Lokalisierungsgenauigkeit kostet, ist mit einem Bild pro Split
aber nicht belastbar zu trennen.

**TTA steht bei YOLO26 nicht zur Verfügung.** `augment=True` liefert bei
`yolo26s-p2` bitidentische Ergebnisse zu `augment=False` (val mAP@0.5 = 0.9400,
test 0.9471 in beiden Fällen) — der end-to-end-Kopf ignoriert das Flag. Der in
Experiment 1d für `yolo11s` gemessene TTA-Gewinn von +2.1 Punkten lässt sich
also nicht auf YOLO26 übertragen und verkleinert dessen Vorsprung entsprechend:
`yolo11s` mit TTA erreicht auf `test` mAP@0.5 = 0.934 gegen 0.947 für
`yolo26s-p2` ohne TTA.

**Kosten.** `yolo26s-p2` braucht 29.9 GB statt 11.6 GB GPU-Speicher und 2.02 h
statt 1.13 h Trainingszeit. Für den Zugewinn von ~5 Punkten mAP@0.5 auf `test`
ist das in dieser Anwendung vertretbar; auf einer kleineren GPU wäre `yolo26s`
(14.5 GB, 0.96 h) der bessere Kompromiss.

### Einschränkung

Auch hier gilt die Aussage aus Kapitel 5.5: das Bootstrap-Konfidenzintervall
für mAP@0.5 beträgt ±0.04 bis ±0.05. Der Abstand `yolo11s` → `yolo26s-p2` liegt
bei nativem Maßstab mit +5.3 Punkten an dessen Grenze, bei val-optimierter
Auflösung mit +1.9 Punkten deutlich darunter. Belastbar ist die Richtung
(`yolo26s-p2` ist in 5 von 6 Metrikspalten auf `test` vorn), nicht die exakte
Differenz.

### Konsequenz für die Arbeit

`yolo11s` bleibt als **dokumentierte Baseline** stehen — die Experimente 1 und 2
wurden darauf gerechnet, und ihre Aussagen (Auflösungsabhängigkeit,
Okklusionsanteil) sind modellunabhängig, weil alle drei Modelle dieselbe
P3-Ebene besitzen. Für einen **produktiven Einsatz** wäre `yolo26s-p2` zu
wählen.

---

# 7 Diskussion

## 7.1 Der Sim-to-Real-Abstand ist messbar und liegt bei etwa 10 Punkten

`sval` (400 synthetische Bilder aus derselben Pipeline, aber mit anderem Seed)
liefert mAP@0.5 = 0.982. Auf echten Bildern sind es 0.886 / 0.888.

**Abstand: −9.6 Prozentpunkte mAP@0.5, −23 Punkte mAP@0.5:0.95.**

Das ist in zwei Richtungen eine gute Nachricht:

* Der Abstand ist **klein genug**, dass das Verfahren praktisch funktioniert.
* Er ist **groß genug**, um zu belegen, dass das Netz *nicht* nur
  Einfügeartefakte gelernt hat — sonst wäre `sval` nahezu perfekt (mAP > 0.99)
  und die realen Bilder katastrophal.

## 7.2 Fehleranalyse

**Falsch-Negative** treten fast ausschließlich auf bei

* sehr kleinen Objekten am linken Bildrand von `test` (12–15 px),
* stark verdeckten Enden tief im Stapel,
* Enden ohne sichtbaren hellen Kranz (vollständig im Schatten).

**Falsch-Positive** sind zum Teil gar keine: mehrere rot markierte Detektionen
im `val`-Bild sitzen auf erkennbaren Rohröffnungen, die in der Ground Truth
fehlen. Bei 88 bzw. 86 handannotierten Objekten schlägt jede einzelne fehlende
Annotation mit ~1.1 Punkten Precision zu Buche. Die berichteten
Precision-Werte sind daher eher eine **Untergrenze**.

Echte Falsch-Positive entstehen an dunklen Löchern im Maschendraht und an
Schattenspalten zwischen Rohren — genau die Strukturen, gegen die die
Negativbilder (9.7 % des Datensatzes) gerichtet sind.

## 7.3 Zusammenfassende Bewertung

Ein ausschließlich auf synthetischen Bildern trainiertes YOLO11s erreicht auf
echten, nie im Training gesehenen Aufnahmen mAP@0.5 ≈ 0.89–0.93 und einen
Zählfehler von 2.3 %. Für eine Zählanwendung im Baustellenkontext ist das
brauchbar.

Mit der besseren Modellwahl aus Experiment 3 (`yolo26s-p2`) steigt das auf
mAP@0.5 = 0.941 (nativ) bzw. 0.947 und F1 = 0.934 auf dem Testbild.

Die Faktoren mit dem größten Hebel sind:

1. **die Inferenzauflösung** (Faktor 2 in mAP@0.5 zwischen 640 und 1600 px),
2. **die physikalische Plausibilität der Synthese** — insbesondere der
   Rohrschaft hinter jeder Stirnfläche, das Tiefenmodell und die Kopplung von
   Objektgröße an den Rohrdurchmesser des Hintergrunds. Diese drei
   Entscheidungen entstanden erst nach visueller Prüfung der Zwischenergebnisse
   und korrigierten jeweils einen physikalisch unmöglichen Bildinhalt.

## 7.4 Einschränkungen und Ausblick

1. **Mehr Evaluationsdaten.** Je ein Bild pro Split ist die größte methodische
   Schwäche der Arbeit. 10–20 annotierte Bilder würden die
   Konfidenzintervalle auf ein Maß bringen, bei dem Experiment 2 entscheidbar
   wäre.
2. **Mehr Cutouts.** Neun Quellobjekte sind die Obergrenze der Formvielfalt.
   Aus den vorhandenen `val`/`test`-Bildern ließen sich mit demselben
   Matting-Verfahren über 150 weitere gewinnen — allerdings nur unter Verzicht
   auf diese Bilder als Evaluationsdaten.
3. ~~**P2-Detektionskopf**~~ — **erledigt**, siehe Experiment 3 (Kapitel 6.3).
   Der Stride-4-Kopf hebt mAP@0.75 auf `test` von 0.320 auf 0.387 und mAP@0.5
   von 0.888 auf 0.941, kostet aber die 2.6-fache GPU-Speichermenge. Die
   verbleibende offene Frage: die mittlere IoU der Treffer verbessert sich
   nicht — die Lokalisierung selbst bleibt also die Schwachstelle, gewonnen
   wird über den Rückruf.
4. **Beleuchtungsvielfalt.** Alle vier Clean Plates sind bei diffusem Licht
   aufgenommen. Harte Schlagschatten, Gegenlicht und Nässe fehlen im
   Quellmaterial und werden nur approximativ durch photometrische
   Augmentierung angenähert.
5. **Kachelweise Inferenz (SAHI)** statt Skalierung des Gesamtbildes. Bei
   deutlich größeren Aufnahmen als 2500 px wäre das die skalierbarere Lösung.
6. **Reflect-Padding erzeugt Spiegelsymmetrie**, wenn ein Hintergrund für den
   gewünschten Zoom zu klein ist (betrifft vor allem `test5_cutted.jpg`).
7. **Der Dellen-Anteil (18 %) ist eine Annahme**, keine Messung.
8. **Die synthetischen Cluster sind im Mittel etwas dichter als real** (13
   statt 10 Nachbarn innerhalb 5 Ø). Bewusst als Robustheitsreserve belassen.

---

# 8 Reproduzierbarkeit

## 8.1 Ablauf

```bash
# 0  Sichtung (erzeugt alle Messwerte aus Kapitel 2)
.venv/bin/python src/analyze_data.py

# 1  Cutouts freistellen  -> assets/masks/, docs/matting_check.png
.venv/bin/python src/matting.py

# 2  Okkluder-Bibliothek  -> assets/bars/, docs/bars_check.png
.venv/bin/python src/occluders.py

# 3  Trainingsdaten       -> synth/train/ + generation_report.json
.venv/bin/python src/generate.py --out synth/train --n 6000 --seed 1 --preview 8
.venv/bin/python src/generate.py --out synth/sval  --n  400 --seed 99

# 4  Training auf GPU 0
.venv/bin/python src/train.py --name baseline

# 5  Auswertung inkl. Ultralytics-Kreuzvergleich und Qualitativbildern
.venv/bin/python src/evaluate.py --weights runs/baseline/weights/best.pt \
    --splits val,test,sval --crosscheck --vis docs/qual_SPLIT_NAME.jpg \
    --out docs/experiments/final_eval.json

# 6  Experimente
.venv/bin/python src/experiment.py infer --weights runs/baseline/weights/best.pt
.venv/bin/python src/experiment.py ablation-data
.venv/bin/python src/experiment.py ablation-train
.venv/bin/python src/experiment.py ablation-eval

# 7  Diagramme und Tabellen
.venv/bin/python src/report.py
```

Anwendung auf eigene Bilder:

```bash
.venv/bin/python src/predict.py MEIN_BILD.jpg          # optional --tta
```

## 8.2 Umgebungshinweis (relevant für die Reproduktion)

In dieser Umgebung (opencv-python 5.0.0.93, torch 2.13, Python 3.10)
verklemmt sich der fork-basierte Ultralytics-DataLoader **reproduzierbar beim
ersten Batch**, sobald `workers > 0` gesetzt ist. Getestet und jeweils
erfolglos:

* `cv2.setNumThreads(0)`,
* `cv2.ocl.setUseOpenCL(False)` und `OPENCV_OPENCL_RUNTIME=disabled`,
* `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
* Multiprocessing-Startmethode `spawn`.

Das Training läuft deshalb mit `workers = 0`. Kompensiert wird das durch
`cache = 'ram'` (die Bilddekodierung entfällt vollständig, gemessener
Zeitgewinn Faktor ~1.6). Erreichter Durchsatz: 3.0 it/s bei Batch 16, GPU zu
etwa 60 % ausgelastet.

Zweiter Stolperstein: Ultralytics stellt einem **relativen** `project`-Pfad
`SETTINGS['runs_dir']` voran. Auf dieser Maschine zeigt das in ein fremdes
Projektverzeichnis. `project` ist deshalb absolut gesetzt.

## 8.3 Verzeichnisstruktur

| Pfad | Inhalt | Umfang |
|---|---|---|
| `src/` | 9 Skripte, 2 885 Zeilen | 176 KB |
| `assets/masks/` | 9 Alphamasken + `ellipses.json` | |
| `assets/bars/` | 56 Okkluder-Stangen | 788 KB gesamt |
| `synth/train/` | 6 000 Bilder + Labels + `generation_report.json` | 3.6 GB |
| `synth/abl_A0/A1/A2/` | Ablationsdatensätze, je 3 000 Bilder | |
| `data/` | val, test, sval + Dataset-YAMLs | 94 MB |
| `runs/baseline/` | Gewichte, `results.csv`, Plots, `parameter_begruendung.md` | 170 MB gesamt |
| `runs/abl_*/` | die drei Ablationsläufe | |
| `docs/` | Dokumentation, Abbildungen, `experiments/*.json` | 16 MB |

## 8.4 Abbildungen

| Datei | Inhalt |
|---|---|
| `docs/matting_check.png` | Ellipsen-Fit über Original (links) und Freistellung (rechts), 12-fach vergrößert |
| `docs/bars_check.png` | die extrahierten Rohrstangen |
| `docs/label_check.png` | Stichprobe synthetischer Labels bei 10-facher Vergrößerung |
| `docs/previews/` | 8 synthetische Trainingsbilder mit eingezeichneten Labels |
| `docs/fig_training.png` | Lernkurven |
| `docs/fig_pr_curves.png` | PR-Kurven und F1 über Konfidenz für val / test / sval |
| `docs/fig_imgsz_sweep.png` | Experiment 1a |
| `docs/fig_ablation.png`, `docs/fig_ablation_beispiele.png` | Experiment 2 |
| `docs/qual_val_*.jpg`, `docs/qual_test_*.jpg` | Detektionen: grün = TP, rot = FP, blau = FN |
