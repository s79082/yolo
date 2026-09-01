# Datensichtung und -bewertung

Alle Zahlen sind reproduzierbar über `src/analyze_data.py`.

---

## 1. Aufgabe und Zielobjekt

Die Bilder zeigen gestapelte **mobile Bauzaunfelder**. Das zu detektierende Objekt
ist die **Stirnfläche eines Rundrohrprofils** – der Kreisring, den man am Ende
eines Rahmenrohres sieht: heller verzinkter Kranz, dunkle Bohrung.

Aus der Sichtung der annotierten Bilder folgt eine Definition, die die ganze
Pipeline prägt:

> Annotiert ist ausschließlich die **sichtbare** Stirnfläche. Rohrschäfte
> (Seitenansicht), Gitterknoten und dunkle Löcher im Maschendraht sind **keine**
> Objekte, obwohl sie ähnlich aussehen.

---

## 2. Cutouts (`/cutouts`, 9 Dateien)

| Datei | Größe | Modus | Bohrung | Ovalität (ry/rx) | Bewertung |
|---|---|---|---|---|---|
| `cutted1.png`  | 38×32 | RGBA | dunkel | 0.86 | brauchbar; Nachbarrohr unten rechts im Ausschnitt |
| `cutted_4.png` | 30×30 | RGBA | hell   | 1.14 | brauchbar; deutlich oval, Kranz einseitig hell |
| `cutted_5.png` | 43×38 | RGBA | dunkel | 0.94 | bestes Exemplar, größte Auflösung |
| `cutted_6.png` | 29×30 | RGBA | hell   | 1.14 | brauchbar; kontrastarm |
| `cutted_7.png` | 41×35 | RGBA | dunkel | 0.87 | sehr sauber, kräftiger Kranz |
| `cutted_8.png` | 39×35 | RGBA | dunkel | 0.83 | **schwächstes Exemplar**: Kranz kontrastarm |
| `cutted_9.png` | 28×30 | RGBA | hell   | 0.97 | brauchbar |
| `cuuted2.png`  | 36×33 | RGBA | dunkel | 0.91 | brauchbar, bräunlicher Farbstich |
| `cuuted3.png`  | 36×34 | RGBA | dunkel | 0.92 | brauchbar |

### Befund 1 — Der Alphakanal ist unbrauchbar (kritisch)

Alle neun Dateien sind zwar RGBA, aber der Alphakanal ist **durchgängig 255**
(gemessen: `alpha.min() == alpha.max() == 255` für alle 9). Es sind rechteckige
Bildausschnitte, keine freigestellten Objekte. Der Anteil der Fläche, der gar
nicht zum Objekt gehört, liegt bei 13–35 %.

**Konsequenz:** Direktes Einkopieren erzeugt an jeder Einfügestelle eine
achsparallele Naht. Ein CNN lernt diese Naht als Abkürzung („shortcut learning“)
statt der Objektform – hervorragende Metriken auf synthetischen Daten,
Totalausfall auf echten. Deshalb ist `src/matting.py` eine eigene Pipelinestufe.

### Befund 2 — Zwei Erscheinungsformen

Sechs Cutouts haben eine **dunkle Bohrung** (offenes Rohr, Blick ins Innere),
drei eine **helle Bohrung** (Sonnenlicht fällt in die Röhre oder das Ende ist
gestaucht/verschlossen). Beide Formen kommen in val/test vor. Die Pipeline
zieht gleichverteilt aus allen neun; eine Gewichtung wäre ohne Häufigkeitszählung
in den Realbildern nicht belegbar.

### Befund 3 — Ovalität ist bereits im Quellmaterial vorhanden

Die aus dem Ellipsen-Fit gewonnenen Achsverhältnisse streuen zwischen 0.83 und
1.14. Ursachen sind Schrägansicht (Kreis → Ellipse) und tatsächlich gequetschte
Rohre. Das rechtfertigt die Ovalisierungs-Transformation der Pipeline.

### Gesamtbewertung Cutouts

**Ausreichend, aber knapp.** Neun Instanzen sind sehr wenig; jede erscheint im
Trainingssatz rund 12 500-mal. Das Risiko, dass das Netz die neun konkreten
Texturmuster auswendig lernt, ist real. Gegenmaßnahmen in der Pipeline:
starke photometrische Streuung pro Instanz, Rotation über den vollen Kreis,
Ovalisierung, Dellen und Perspektive – und bewusst **kein** großes Modell.

---

## 3. Hintergründe (`/bg`, 7 Dateien)

| Datei | Größe | Typ | Rohr‑Ø | Gewicht | Bewertung |
|---|---|---|---|---|---|
| `test_cutted.png`  | 1513×959  | **Clean Plate** | 16 px | 6 | Stapelwand, exakt die Zieldomäne |
| `test2_cutted.jpg` | 1706×1108 | **Clean Plate** | 18 px | 6 | Stapelwand + Gittermatten + Laub |
| `test4_cutted.png` | 1517×2788 | **Clean Plate** | 21 px | 6 | größte Fläche, hochkant, Laub am Boden |
| `test5_cutted.jpg` | 945×552   | **Clean Plate** | 20 px | 6 | Aufsicht, Absperrband, Gras |
| `bg_3.jpg` | 1024×576 | Fremdszene | – | 1 | Kieswerk, Schotter, Himmel |
| `bg_4.jpg` | 800×600  | Fremdszene | – | 1 | Betonplatten, Sandsäcke, Personen |
| `bg_5.jpg` | 800×600  | Fremdszene | – | 1 | Betondeck + **Maschendrahtzaun** |

### Befund 4 — Vier echte Clean Plates (der wertvollste Teil des Materials)

`test*_cutted.*` sind Aufnahmen derselben Objektart, aus denen die Stirnflächen
herausgeschnitten wurden: Wände aus Rohrschäften **ohne** Stirnflächen. Sie
liefern

1. den photometrisch exakt passenden Kontext für das Einkopieren,
2. die **Okkluder-Bibliothek** (`src/occluders.py` schneidet daraus 56 echte
   Rohrstangen samt korrekter zylindrischer Schattierung),
3. echte Negativbeispiele.

Sie erhalten deshalb 6-faches Ziehgewicht.

### Befund 5 — Die drei Fremdszenen sind hartes Negativmaterial

`bg_5.jpg` enthält einen Maschendrahtzaun mit rautenförmigen Öffnungen – genau
die Struktur, die als Falschpositiv droht. `bg_3.jpg` liefert Schotter (viele
dunkle, runde Kiesel). Diese Bilder sind primär als **Negativbilder** wertvoll,
nicht als Objektträger.

### Befund 6 — Der Rohrdurchmesser verankert den Maßstab

Aus den Clean Plates lässt sich der Rohrdurchmesser messen (16/18/21/20 px). Eine
Stirnfläche hat zwangsläufig **denselben** Durchmesser wie der Schaft daneben.
Der Generator koppelt deshalb Zoomfaktor und Objektgröße aneinander, statt beide
unabhängig zu ziehen.

### Gesamtbewertung Hintergründe

**Gut.** Die vier Clean Plates sind qualitativ genau das Richtige. Schwäche:
alle vier zeigen dieselbe Stapelart bei bedecktem bis diffusem Licht — kein
harter Sonnenschein, kein Nass/Schnee, kein Nachtbild. Diese Lücke schließt die
Pipeline nur teilweise über photometrische Augmentierung; sie bleibt die
größte verbleibende Domänenlücke.

---

## 4. Validierungs- und Testdaten

|  | Bild | Auflösung | Objekte |
|---|---|---|---|
| `val`  | `PXL_20251204_073853552.jpg` | 2459×1174 | 88 |
| `test` | `test5.jpg`                  | 1600×900  | 86 |

### Gemessene Objektstatistik

| Größe | val | test |
|---|---|---|
| Breite (px) | 19.9 – 38.5, Median 29.3 | 9.3 – 33.3, Median 23.8 |
| Höhe (px) | 15.7 – 34.2, Median 27.0 | 6.4 – 30.5, Median 19.7 |
| √Fläche (px) | 20.6 – 35.8, **Median 27.4** | 12.2 – 31.9, **Median 19.6** |
| Seitenverhältnis w/h | p10 0.97, Median 1.08, p90 1.31 | p10 0.85, Median 1.09, p90 1.22 |
| Objektfläche gesamt | 2.48 % des Bildes | 2.69 % |
| angeschnitten am Rand | 0 % | 5.8 % |

### Gemessene Anordnungsstatistik (n = 174)

| Größe | val | test |
|---|---|---|
| Nächster-Nachbar-Abstand (in Ø) | p10 0.86, Median 1.09, p90 1.78 | p10 0.83, Median 1.24, p90 2.51 |
| Nachbarn innerhalb 5 Ø | Median 10, max 14 | Median 10, max 20 |
| Objekte, die ihren nächsten Nachbarn berühren (< 1.2 Ø) | 61.4 % | 46.5 % |
| Größenverhältnis benachbarter Objekte | Median 1.05, p90 1.14 | Median 1.05, p90 1.17 |
| Nachbarrichtung (0° = horizontal) | quasi isotrop, leichte Häufung 90–150° | quasi isotrop |

### Befund 7 — Sehr kleine Objekte, dicht gepackt

Bei 20–36 px Kantenlänge in Bildern von 1600–2459 px Breite sind das
Kleinobjekte im COCO-Sinne. Die Konsequenz für die Inferenz ist zwingend:
bei `imgsz=640` schrumpfen sie auf 5–9 px und liegen unter der Auflösung des
feinsten Detektionskopfes (Stride 8). Deshalb wird bei **nativem Maßstab**
inferiert und `imgsz` in Experiment 1 systematisch untersucht.

### Befund 8 — Nachbarn haben (fast) dieselbe Größe

Das Größenverhältnis benachbarter Objekte liegt im Median bei 1.05. Innerhalb
eines Stapels haben alle Rohre dieselbe Nennweite. Der Generator zieht deshalb
**eine Basisgröße pro Cluster** und variiert je Instanz nur um ±8 % — statt
jede Instanz unabhängig zu würfeln, was ein sofort erkennbarer Realismusfehler
wäre.

### Befund 9 — Jede Stirnfläche hat einen Schaft (die wichtigste Randbedingung)

In den Nahaufnahmen (`docs/`-Vergleichsbilder) gilt ausnahmslos: hinter jeder
annotierten Stirnfläche führt ein sichtbarer Rohrschaft weg. Ein Kreis mitten
auf einem durchgehenden Rohr ist physikalisch unmöglich. Der Generator setzt
deshalb bei 88 % der Instanzen einen echten Rohrschaft **hinter** das Profil.

### Befund 10 — Datenlage der Evaluation ist die größte methodische Schwäche

**Je ein Bild** für Validierung und Test. Das ist für eine belastbare
Metrikangabe zu wenig: die Streuung einer mAP-Schätzung auf 88 Objekten ist
groß, und beide Bilder zeigen dieselbe Objektart bei ähnlichem Wetter. Alle
berichteten Zahlen sind entsprechend als **Punktschätzung mit erheblicher
Unsicherheit** zu lesen. Das Test-Framework gibt deshalb zusätzlich
Bootstrap-Konfidenzintervalle aus.
