# PipeNet verständlich erklärt

*Zielgruppe: Informatikstudierende. Vorausgesetzt werden Programmierkenntnisse
und Grundlagen der linearen Algebra. **Nicht** vorausgesetzt werden Kenntnisse
in Computer Vision oder Deep Learning — jeder Begriff wird bei seiner ersten
Verwendung eingeführt.*

Die knappe, formale Fassung mit allen Messwerten steht in
[`docs/ARCHITEKTUR.md`](docs/ARCHITEKTUR.md). Dieses Dokument erklärt
stattdessen **warum** die Architektur so aussieht, wie sie aussieht.

![PipeNet Architektur](docs/fig_architektur.png)

---

## Inhalt

1. [Das Problem als Rechenaufgabe](#1-das-problem-als-rechenaufgabe)
2. [Warum klassische Bildverarbeitung hier scheitert](#2-warum-klassische-bildverarbeitung-hier-scheitert)
3. [Die CNN-Grundlagen, die man braucht](#3-die-cnn-grundlagen-die-man-braucht)
4. [Das eigentliche Problem: variable Ausgabelänge](#4-das-eigentliche-problem-variable-ausgabelänge)
5. [PipeNet Schicht für Schicht](#5-pipenet-schicht-für-schicht)
6. [Zielkodierung: vom Label zum Tensor](#6-zielkodierung-vom-label-zum-tensor)
7. [Die Verlustfunktion](#7-die-verlustfunktion)
8. [Dekodierung: vom Tensor zurück zur Liste](#8-dekodierung-vom-tensor-zurück-zur-liste)
9. [Training: was in der Praxis wehtut](#9-training-was-in-der-praxis-wehtut)
10. [Evaluation: was mAP eigentlich misst](#10-evaluation-was-map-eigentlich-misst)
11. [Was der Entwurf gebracht hat — und was nicht](#11-was-der-entwurf-gebracht-hat--und-was-nicht)
12. [Glossar](#12-glossar)

---

## 1 Das Problem als Rechenaufgabe

**Anwendung.** Auf Fotos von gestapelten Bauzaunfeldern sollen die
**Stirnflächen der Rahmenrohre** gefunden und gezählt werden — die dunklen
Ringe, die man sieht, wenn man von vorne in ein Rohrbündel schaut.

**Signatur.** Formal ist das eine Funktion

```
f : R^(H × W × 3)  →  { (x, y, w, h, s) }*
```

Ein Bild rein, eine **Menge** von Boxen mit Konfidenz raus. Die Menge hat
**keine feste Länge** — auf einem Foto sind es 3 Objekte, auf dem nächsten 90.

Das ist der Punkt, an dem sich Detektion von Klassifikation unterscheidet.
Ein Klassifikator bildet auf einen Vektor fester Länge ab (`softmax` über *k*
Klassen); das kann ein Netz direkt ausgeben. Eine Menge variabler Länge kann
es nicht. Wie man dieses Problem löst, ist die zentrale Entwurfsentscheidung
jedes Detektors — siehe Abschnitt 4.

**Die gemessenen Eigenschaften dieser Instanz.** Sie bestimmen alles Weitere
(Herleitung in `../docs/DATENSICHTUNG.md`):

| Eigenschaft | Wert | Konsequenz |
|---|---|---|
| Klassen | 1 | kein Klassifikationskopf nötig |
| Objektdurchmesser | 12–44 px, Median 20–27 px | Kleinobjekt-Regime |
| Größenspanne | Faktor 3.7 | eine Auflösungsebene genügt |
| Objekte je Bild | bis 90 | dichte Vorhersage sinnvoll |
| Abstand zum nächsten Nachbarn | Median 1.09–1.24 Durchmesser | **Boxen überlappen, Zentren nicht** |
| Objekte, die ihren Nachbarn berühren | 46–61 % | NMS problematisch |
| Trainingsmaterial | 9 ausgeschnittene Objekte | Überanpassungsgefahr |

---

## 2 Warum klassische Bildverarbeitung hier scheitert

Die naheliegende Frage: die Objekte sind Kreise — warum nicht einfach
`cv2.HoughCircles`?

Weil die Kreis-*Geometrie* nicht das Problem ist, sondern die
**Unterscheidung von Ähnlichem**. Auf denselben Bildern gibt es:

* dunkle Löcher im Maschendrahtgitter (rautenförmig, aber ähnlich groß),
* Schattenspalten zwischen Rohren,
* Kiesel und Laub am Boden,
* Rohre in Seitenansicht (Zylinder, keine Ringe) — die häufigste Struktur im
  Bild überhaupt.

Ein Hough-Detektor arbeitet auf Kantenbildern und kennt nur „rund/nicht rund".
Er kann nicht unterscheiden, ob ein runder dunkler Fleck ein Rohrende, ein
Schatten oder ein Stein ist. Dafür braucht man **Textur- und Kontextwissen**:
ein Rohrende hat einen hellen verzinkten Kranz, eine dunkle Bohrung, und
dahinter führt ein Rohrschaft weg.

Genau solches Wissen kann man nicht sinnvoll als Regelsystem hinschreiben —
aber man kann es aus Beispielen lernen lassen. Das ist die Begründung für ein
CNN, nicht „weil man das heute so macht".

---

## 3 Die CNN-Grundlagen, die man braucht

Falls du damit schon vertraut bist: weiter zu Abschnitt 4.

### 3.1 Faltung als gelernter lokaler Filter

Eine **Faltung** (convolution) schiebt eine kleine Gewichtsmatrix — den
**Kernel**, typisch 3×3 — über das Bild und berechnet an jeder Position das
Skalarprodukt aus Kernel und Bildausschnitt:

```
out[y, x] = Σ_{i,j}  kernel[i, j] · in[y + i, x + j]
```

Das ist derselbe Mechanismus wie ein Sobel- oder Gauß-Filter in der
klassischen Bildverarbeitung — mit einem Unterschied: **die Kernelwerte sind
nicht vorgegeben, sondern werden gelernt.**

Zwei Eigenschaften folgen daraus, und beide sind der Grund, warum CNNs für
Bilder funktionieren:

* **Gewichtsteilung.** Derselbe Kernel wird an allen Positionen angewandt. Ein
  Kantendetektor muss nicht 2 Millionen Mal separat gelernt werden, sondern
  einmal. Ein voll verbundenes Netz auf einem 1024²-Bild hätte pro Neuron über
  3 Millionen Gewichte; ein 3×3-Kernel hat 9.
* **Translationsäquivarianz.** Verschiebt man das Eingangsbild, verschiebt sich
  die Ausgabe genauso. Ein Rohrende wird links oben genauso erkannt wie rechts
  unten.

### 3.2 Kanäle und Merkmalskarten

Ein Layer wendet nicht einen Kernel an, sondern *C* viele. Die Ausgabe ist
deshalb ein Tensor der Form `(C, H, W)` — *C* **Merkmalskarten**
(feature maps), je eine pro Kernel. Jede Karte antwortet auf ein anderes
Muster: eine auf senkrechte Kanten, eine auf Helligkeitssprünge, später eine
auf Ringstrukturen.

`C` heißt **Kanalzahl**. In PipeNet: Eingang 3 (RGB), dann 64, 128, 256.

### 3.3 Stride und Auflösungsverlust

Mit `stride=2` springt der Kernel in Zweierschritten. Die Ausgabekarte ist
dann halb so groß. Das macht man aus zwei Gründen:

1. **Rechenaufwand.** Halbe Auflösung = ein Viertel der Positionen.
2. **Rezeptives Feld.** Nach zwei Stride-2-Schritten „sieht" ein 3×3-Kernel
   effektiv einen 12×12-Bereich des Originalbildes. Nur so kann ein Netz
   größere Zusammenhänge erfassen.

Der **Stride** eines Layers ist der Gesamtfaktor gegenüber dem Eingang. Bei
Stride 16 ist die Merkmalskarte 16-mal kleiner als das Bild. Man schreibt
`/16`. Die Ebenen heißen konventionell `P2` (Stride 4), `P3` (8), `P4` (16),
`P5` (32) — `Pn` steht für Stride 2ⁿ.

> **Das ist die Zahl, um die sich in diesem Projekt fast alles dreht.** Ein
> Objekt von 22 px belegt auf einer Karte mit Stride *s* genau 22/*s* Zellen.
> Bei Stride 4 sind das 5.5 Zellen, bei Stride 16 nur 1.4. Eine Merkmalskarte
> mit Stride 16 kann ein 22-px-Objekt nicht mehr *lokalisieren* — es passt in
> eine einzige Zelle.

### 3.4 Was „Training" bedeutet

Das Netz hat *N* Gewichte (bei PipeNet N = 3 070 917). Training heißt:

1. **Verlustfunktion** definieren, die misst, wie falsch die Ausgabe ist:
   `L(θ)`, wobei θ alle Gewichte sind.
2. Ein Batch Bilder durchschicken (**Forward Pass**), `L` berechnen.
3. Per Kettenregel `∂L/∂θ` berechnen (**Backpropagation**) — automatisch von
   PyTorch, weil jede Operation ihre Ableitung kennt.
4. Gewichte gegen den Gradienten verschieben: `θ ← θ − η · ∂L/∂θ`.
5. Millionen Mal wiederholen.

`η` ist die **Lernrate**. Der Optimierer (hier **AdamW**) verwaltet zusätzlich
laufende Mittelwerte der Gradienten, damit die Schrittweite pro Gewicht
automatisch skaliert.

### 3.5 Transfer Learning

Ein von Grund auf trainiertes Netz muss auch die trivialen Filter erst lernen
— Kanten, Farbübergänge, Ecken. Diese sind aber **domänenunabhängig**: ein
Kantendetektor sieht bei Katzenbildern genauso aus wie bei Rohren.

Deshalb startet PipeNet mit einem **ResNet18**, das auf ImageNet (1.2 Mio.
Bilder, 1000 Klassen) vortrainiert wurde. Nur die oberen Schichten müssen sich
an die neue Aufgabe anpassen. Das ist hier besonders wichtig, weil wir nur
9 echte Objektinstanzen haben — je mehr Gewichte aus einer fremden, großen
Datenquelle kommen, desto weniger kann sich das Netz auf unsere 9 Texturen
festlegen.

---

## 4 Das eigentliche Problem: variable Ausgabelänge

Zurück zu Abschnitt 1: ein Netz gibt Tensoren fester Form aus, gebraucht wird
eine Menge variabler Länge. Es gibt zwei etablierte Auswege.

### 4.1 Ansatz A — dichtes Boxraster plus NMS (YOLO, SSD, Faster R-CNN)

Man legt ein Raster über das Bild und lässt das Netz **für jede Rasterzelle**
vorhersagen: „liegt hier ein Objekt, und wenn ja, wie ist seine Box relativ zu
dieser Zelle?" Bei einem 240×240-Raster über drei Auflösungsebenen sind das
über 76 000 Kandidatenboxen — **feste** Ausgabeform, Problem gelöst.

Der Preis: ein Objekt löst typischerweise mehrere benachbarte Zellen aus. Man
erhält also Duplikate und muss sie nachträglich entfernen. Das macht
**Non-Maximum Suppression (NMS)**:

```
sortiere Kandidaten nach Konfidenz absteigend
für jeden Kandidaten K (bester zuerst):
    behalte K
    verwirf alle noch offenen Kandidaten mit IoU(K, ·) > τ
```

**IoU** (Intersection over Union) ist das Standardmaß für Boxüberlappung:
Schnittfläche geteilt durch Vereinigungsfläche, also 1.0 bei Deckungsgleichheit
und 0.0 bei Disjunktheit.

Und hier liegt das Problem für **unsere** Daten. τ ist ein freier Parameter,
und die Wahl ist ein echter Kompromiss:

* τ zu klein → zwei *verschiedene*, dicht benachbarte Rohre werden für
  Duplikate gehalten. Ein echtes Objekt geht verloren.
* τ zu groß → Duplikate bleiben stehen. Falschpositive.

Bei gut getrennten Objekten (Autos, Personen) ist τ = 0.5…0.7 unkritisch. Bei
uns liegt der Abstand benachbarter Zentren im 10-%-Quantil bei **0.84
Objektdurchmessern** — die Boxen überlappen deutlich. Es gibt kein τ, das
beide Fehler gleichzeitig vermeidet.

### 4.2 Ansatz B — Zentrums-Heatmap (CenterNet, und PipeNet)

Die andere Sicht: statt „Boxen mit Duplikatunterdrückung" formuliert man die
Aufgabe als **dichte Regression eines Bildes**.

Das Netz gibt eine einkanalige Karte `hm` aus, gleich groß wie eine
Merkmalskarte bei Stride 4. `hm[y, x] ∈ [0,1]` bedeutet: *„wie sicher bin ich,
dass hier das Zentrum eines Objekts liegt?"* Feste Ausgabeform — auch hier
Problem gelöst, aber anders.

Zurück in eine Menge kommt man durch **Extraktion lokaler Maxima**: jede
Zelle, die größer ist als alle 8 Nachbarn, ist ein Objekt.

**Warum das hier passt.** Die entscheidende Messung: benachbarte *Boxen*
überlappen, aber benachbarte *Zentren* sind immer getrennt — minimal 0.83
Durchmesser ≈ 16 px ≈ **4 Zellen** bei Stride 4. Ein lokales Maximum in einem
3×3-Fenster braucht nur **2 Zellen** Abstand, um von einem anderen getrennt zu
bleiben. Der Sicherheitsabstand ist also Faktor 2.

Die Konsequenz ist bemerkenswert: **es gibt keine IoU-Schwelle mehr.** Der
Parameter τ existiert nicht. Statt eines Kompromisses hat man eine harte,
überprüfbare Bedingung („Zentren ≥ 2 Zellen auseinander"), und diese Bedingung
ist in den vorliegenden Daten mit Faktor 2 Reserve erfüllt.

| | Ansatz A (Box + NMS) | Ansatz B (Heatmap + Peaks) |
|---|---|---|
| Ausgabe | Boxen je Rasterzelle | 1-Kanal-Wahrscheinlichkeitskarte |
| Mengenbildung | NMS über IoU | lokale Maxima |
| freier Parameter | τ (IoU-Schwelle) | **keiner** |
| Fehlermodus | Nachbar verloren / Duplikat | zwei Zentren < 2 Zellen verschmelzen |
| tritt hier auf? | ja, unvermeidbar | nein (Minimum 4 Zellen) |

---

## 5 PipeNet Schicht für Schicht

Konkrete Tensorformen für einen 1024×1024-Eingang. Auslesen kannst du sie mit
`python cnn/src/model.py`.

### 5.1 Encoder — Auflösung runter, Semantik rauf

```
Eingang                     (3, 1024, 1024)
 stem     conv 7×7, s2  →  (64,  512,  512)   /2      9.5 k Par.
 maxpool  s2            →  (64,  256,  256)   /4
 layer1                 →  (64,  256,  256)   /4    148 k
 layer2                 →  (128, 128,  128)   /8    526 k
 layer3                 →  (256,  64,   64)   /16  2 100 k
 layer4                 →  (512,  32,   32)   /32  8 394 k   ← NICHT VERWENDET
```

`layer1…layer4` sind die Stufen des ResNet18. Jede besteht aus zwei
**Residualblöcken** — zwei 3×3-Faltungen plus eine Abkürzungsverbindung
(`out = f(x) + x`). Die Abkürzung ist der Grund, warum sich tiefe Netze
überhaupt trainieren lassen: der Gradient kann durch sie ungehindert
zurückfließen, statt über viele Layer hinweg zu verschwinden.

**Entscheidung 1: `layer4` fällt weg.** Es enthält 8.39 M der 11.69 M
ResNet18-Parameter — 72 % — und arbeitet auf Stride 32. Bei einem 22-px-Objekt
sind das 0.7 Zellen; die Ebene kann das Objekt nicht mehr auflösen. Das
rezeptive Feld am Ende von `layer3` ist bereits deutlich größer als jedes
Objekt samt Umgebung. Weglassen kostet also nichts an Kontext und spart 72 %
des Encoders.

Und es ist nicht nur Sparsamkeit: bei **9** zugrunde liegenden
Objektinstanzen, von denen jede rund 15 000 Mal im Trainingssatz erscheint,
ist überschüssige Kapazität gleichbedeutend mit **Auswendiglernen**. Ein
kleineres Modell ist hier nicht Kompromiss, sondern Regularisierung.

### 5.2 Dekoder — Auflösung wieder hoch

Der Encoder hat ein Dilemma erzeugt: `layer3` weiß *was* im Bild ist, kann
aber nicht mehr genau sagen *wo* (64×64 Zellen für ein 1024²-Bild). `layer1`
weiß genau wo, hat aber kaum Kontext.

Die Lösung ist eine **Feature Pyramid** (FPN): von oben nach unten
hochskalieren und mit den feineren Encoder-Karten addieren.

```
p3 = lat3(c3)                              # 1×1: 256 → 64 Kanäle,  /16
p2 = smooth2( lat2(c2) + upsample(p3) )    #                        /8
p1 = smooth1( lat1(c1) + upsample(p2) )    #                        /4
f  = fuse( p1 + fine(stem_out) )           #                        /4
```

Die `lat*`-Faltungen sind 1×1 und dienen nur dazu, alle Ebenen auf 64 Kanäle zu
bringen, damit man sie addieren kann. `upsample` ist Nearest-Neighbor-
Verdopplung. Die `smooth*`-Faltungen (3×3) glätten die Additionsartefakte.

**Entscheidung 2: nur eine Detektionsebene, bei Stride 4.** YOLO detektiert auf
P3/P4/P5 gleichzeitig und lässt jede Ebene einen Größenbereich abdecken. Das
ist richtig, wenn Objektgrößen um zwei Größenordnungen streuen. Hier streuen
sie um **Faktor 3.7** (12–44 px) — das schafft eine Ebene, wenn der
Größenkopf (Abschnitt 5.3) den Rest übernimmt. Zwei der vier YOLO-Ebenen wären
für diese Objektgrößen ohnehin unbrauchbar.

**Entscheidung 3: ein zusätzlicher Skip von Stride 2.** Das diskriminative
Merkmal ist der **1–3 px breite helle Rohrkranz**. `stem` (Stride 2) führt ihn
noch, `layer1` (nach MaxPool) nur abgeschwächt. `fine` ist eine 3×3-Faltung mit
Stride 2, die den Stem-Ausgang auf Stride 4 bringt; sie wird addiert. Kosten:
37 k Parameter.

### 5.3 Die drei Köpfe

Ein **Kopf** (head) ist ein kleines Faltungspaar, das die gemeinsame
Merkmalskarte `f` auf die gewünschte Ausgabe abbildet: 3×3 mit 64 Kanälen,
dann 1×1 auf die Zielkanalzahl.

```
hm  : (1, 256, 256)   Zentrumswahrscheinlichkeit (Logit)
wh  : (2, 256, 256)   log(Breite), log(Höhe) in Bildpixeln
off : (2, 256, 256)   Subpixel-Versatz in Zelleinheiten, [0,1)
```

**Warum `off` existiert.** Ohne diesen Kopf wäre die Zentrumsvorhersage auf das
Stride-4-Raster quantisiert. Der Fehler beträgt dann 0…4 px **pro Achse**, im
Mittel 2 px. Für eine quadratische Box der Kantenlänge *d*, um Δ in beiden
Achsen verschoben, gilt

```
IoU(Δ) = (d − Δ)² / (2d² − (d − Δ)²)
```

Eingesetzt:

| d | Δ = 2 px (Mittel) | Δ = 4 px (Grenzfall) |
|---|---|---|
| 20 px | IoU 0.68 | IoU 0.47 |
| 27 px | IoU 0.75 | IoU 0.57 |

Ein 20-px-Objekt verliert also im Mittel **0.32 IoU allein durch
Quantisierung** — bei einer Bewertungsschwelle von 0.5 ist das die Differenz
zwischen Treffer und Fehlschlag. Der Versatzkopf lernt genau diesen fehlenden
Bruchteil und drückt den gemessenen systematischen Zentrumsfehler auf **unter
einen Pixel**.

**Warum `wh` logarithmisch.** Ein L1-Verlust auf `log w` bestraft den
*relativen* Fehler: `|log w_pred − log w_gt| = |log(w_pred/w_gt)|`. In
Pixeleinheiten wären 2 px Fehler bei einem 44-px-Objekt (4.5 %) und bei einem
12-px-Objekt (17 %) gleich teuer, obwohl der zweite Fall viel schlimmer ist.

**Parameterbudget.**

| Block | Parameter | Anteil |
|---|---|---|
| `layer3` | 2 099 712 | 68.4 % |
| `layer2` | 525 568 | 17.1 % |
| `layer1` | 147 968 | 4.8 % |
| Köpfe (3×) | 111 301 | 3.6 % |
| Dekoder (`lat*`, `smooth*`, `fine`, `fuse`) | 176 832 | 5.8 % |
| `stem` | 9 536 | 0.3 % |
| **Gesamt** | **3 070 917** | |

Zum Vergleich: `yolo11s` 9.43 M, `yolo26s` 9.95 M, `yolo26s-p2` 9.66 M.

---

## 6 Zielkodierung: vom Label zum Tensor

Die Labels liegen als Boxen vor (`cx, cy, w, h` in Pixeln). Das Netz gibt aber
Tensoren aus. Für den Verlust muss man die Labels also in **exakt dieselbe
Form** bringen. Das ist der Teil, der beim ersten Kontakt am wenigsten
offensichtlich ist.

```python
for cx, cy, bw, bh in box:
    ox, oy = cx / stride, cy / stride          # Zentrum in Zellkoordinaten
    ix, iy = int(ox), int(oy)                  # zuständige Zelle
    r     = gaussian_radius(bh / stride, bw / stride, min_overlap=0.7)
    sigma = clip((2 * max(1.0, r) + 1) / 6.0, 0.6, 2.5)
    draw_gaussian(hm, ix, iy, sigma)           # weiche Glocke
    hm[iy, ix]      = 1.0                      # exaktes Zentrum: hartes Positiv
    wh[0, iy, ix]   = log(bw)
    wh[1, iy, ix]   = log(bh)
    off[0, iy, ix]  = ox - ix                  # Rest nach der Ganzzahldivision
    off[1, iy, ix]  = oy - iy
    mask[iy, ix]    = 1.0                      # markiert die Zellen mit Label
```

### Warum eine Gauß-Glocke und nicht ein einzelnes 1?

Wäre das Ziel eine einzige 1 in einem Meer von Nullen, würde das Netz für eine
Vorhersage **einen Pixel neben** dem Zentrum genauso hart bestraft wie für eine
Vorhersage am ganz falschen Bildrand. Das ist erstens unfair und zweitens
schlecht für den Gradienten: das Optimierungsproblem wird unnötig scharf.

Die Glocke gibt eine **räumliche Toleranz**: nahe am Zentrum ist der Zielwert
fast 1, weiter draußen fällt er ab.

### Woher kommt σ?

Nicht geraten, sondern aus einer geometrischen Bedingung abgeleitet
(CornerNet, Gl. 6). `min_overlap = 0.7` bedeutet: *wie weit darf eine Box
verschoben werden, damit sie mit dem Original noch 0.7 IoU hat?* Diese Distanz
ist der Radius *r*; daraus folgt σ = (2r+1)/6 (drei σ decken den Radius ab).

Für unsere Größen ergibt das:

| Objekt | Zellen (Stride 4) | Radius | σ |
|---|---|---|---|
| 12 px | 3.0 | 0.82 | 0.60 |
| 20 px | 5.0 | 1.37 | 0.62 |
| 27 px | 6.8 | 1.84 | 0.78 |
| 44 px | 11.0 | 3.01 | 1.17 |

σ liegt also bei 0.6–1.2 Zellen, der Nachbarabstand bei 4.4–5.5 Zellen. **Die
Glocken benachbarter Objekte überlappen nicht** — genau die Voraussetzung
dafür, dass die Peak-Extraktion sie später trennen kann. Wäre σ zu groß,
würden zwei Objekte zu einem Plateau verschmelzen und die ganze Konstruktion
fiele zusammen.

Eingezeichnet wird mit `np.maximum`, **nicht** additiv. Bei sich berührenden
Objekten würde eine Summe zwischen zwei echten Zentren Werte > 1 erzeugen —
und damit ein künstliches drittes Maximum.

---

## 7 Die Verlustfunktion

Drei Terme: `L = L_hm + 1.0 · L_wh + 1.0 · L_off`.

### 7.1 Das Klassenungleichgewicht

In einem 256×256-Ausgaberaster mit ~23 Objekten sind **65 513 von 65 536**
Zellen negativ — 99.96 %. Mit gewöhnlicher binärer Kreuzentropie würde die
Summe der Negativterme die 23 Positivterme vollständig erschlagen. Das Netz
lernt in wenigen Schritten die triviale Lösung „überall 0" und bleibt dort.

Gegenmittel ist der **Gaussian Focal Loss** aus CornerNet/CenterNet:

```python
p = sigmoid(logits)
pos_loss = -((1 - p) ** α) * log(p)                    * [target == 1]
neg_loss = -((1 - target) ** β) * (p ** α) * log(1 - p) * [target != 1]
L_hm     = (pos_loss.sum() + neg_loss.sum()) / max(1, n_pos)
```

Die beiden Exponenten (α = 2, β = 4) tun verschiedene Dinge:

* **α** dämpft *leichte* Beispiele. Eine Negativzelle, die schon korrekt bei
  p = 0.01 liegt, trägt mit 0.01² = 0.0001 praktisch nichts mehr bei. Der
  Gradient konzentriert sich automatisch auf die schwierigen Fälle.
* **β** dämpft Negative **in der Nähe eines Zentrums**, wo das gauss-verschmierte
  Ziel ohnehin nahe 1 liegt. Ohne diesen Term würde das Netz dafür bestraft,
  den Nachbarpixel eines Zentrums ebenfalls hoch zu bewerten — also genau für
  die Toleranz, die man bei 5.5 Zellen großen Objekten haben will.

Normiert wird durch `n_pos`, die Anzahl echter Zentren — nicht durch die
Zellzahl. Sonst hinge die Verlustgröße von der Bildgröße ab und die Lernrate
wäre nicht übertragbar.

**Bias-Initialisierung.** Der letzte 1×1-Layer des `hm`-Kopfes startet mit
Bias `−4.595 = logit(0.01)`. Das Netz sagt initial überall „kein Objekt, p =
0.01" voraus. Ohne diesen Prior sind die ersten Iterationen ein steiler
Abstieg von p ≈ 0.5 auf p ≈ 0 über 65 000 Zellen — verschwendete Schritte, und
im schlechten Fall kollabiert der Kopf.

### 7.2 Maskierte Regression

`wh` und `off` sind nur an den 23 Zellen mit Label **definiert**. An allen
anderen gibt es keinen sinnvollen Zielwert. Also wird der L1-Verlust dort
maskiert:

```python
m = mask.bool().unsqueeze(1).expand_as(pred)
L = F.l1_loss(pred[m], target[m], reduction="sum") / mask.sum()
```

Ohne die Maske würde das Netz gezwungen, überall *irgendeine* Größe
vorherzusagen, und der Gradient der 99.96 % undefinierten Zellen würde das
Signal der 0.04 % definierten erschlagen — dasselbe Problem wie in 7.1, nur an
anderer Stelle.

**Die Gewichte 1.0/1.0** ergeben sich aus den Größenordnungen: der L1-Fehler
auf log-Größen liegt bei 0.05–0.3, der auf Versätzen bei 0.1–0.3, und `L_hm`
in derselben Region. CenterNet verwendet 0.1 für `wh` — das gilt aber für
Größen in **Pixeleinheiten**, wo der Fehler bei 2–5 liegt.

---

## 8 Dekodierung: vom Tensor zurück zur Liste

```python
keep = (hm == F.max_pool2d(hm, 3, stride=1, padding=1))   # lokale Maxima
h    = hm * keep
score, idx = h.view(1, -1).topk(k)
iy, ix = idx // W, idx % W
```

Der `max_pool2d`-Trick ist der Kern: ein 3×3-MaxPool mit `stride=1` ersetzt
jede Zelle durch das Maximum ihrer 3×3-Nachbarschaft. Wo Ausgabe == Eingabe
ist, war die Zelle bereits das Maximum ihrer Nachbarschaft. Ein Vergleich, kein
Sortieren, kein Schleifendurchlauf — **eine** GPU-Operation für das, was NMS
mit einer Schleife über sortierte Kandidaten macht.

Dann werden Größe und Versatz an genau diesen Positionen ausgelesen und die
Box rekonstruiert:

```python
cx = (ix + off[0]) * stride       # Subpixel-Rekonstruktion
cy = (iy + off[1]) * stride
w, h = wh[0].exp(), wh[1].exp()   # log-Größe umkehren
box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
```

### Koordinatenkonvention — die Stolperfalle

Bei Stride > 1 muss festgelegt werden, welchem Eingangspixel eine Ausgabezelle
entspricht. PipeNet verwendet durchgängig **Zelle *j* ↔ Pixel 4*j*** (Blockanfang),
weil Zielkodierung (`int(ox)`) und Dekodierung (`(ix + off) * 4`) exakt so
zueinander passen. Gemessen liegt der systematische Zentrumsversatz der
Vorhersagen gegenüber der Ground Truth deshalb bei **unter einem Pixel**.

Die Heatmap-*Visualisierung* hatte trotzdem einen Fehler: ein naives
`cv2.resize(hm, (W, H))` streckt das /32-Padding mit und zeichnet Zellen am
Blockanfang statt in der Blockmitte — zusammen bis zu 7.7 px Versatz, bei
20-px-Objekten also 39 %. Details in
[`docs/ARCHITEKTUR.md`](docs/ARCHITEKTUR.md) Abschnitt 7b.

Der Fall ist lehrreich: **ein Darstellungsfehler sieht einem Modellfehler zum
Verwechseln ähnlich.** Auseinanderhalten ließen sich beide nur durch Messung
des Versatzes gegen die Ground Truth, nicht durch Hinsehen.

### Test-Time-Augmentation über D4

Rohrenden sind nahezu rotationssymmetrisch. Drehungen um 90° und Spiegelungen
sind auf einem Pixelgitter **exakt** (reine Indexpermutationen, keine
Interpolation) — anders als die übliche Mehrskalen-TTA, die immer resampelt.
PipeNet kann deshalb über alle 8 Elemente der **Diedergruppe D4** mitteln:
Bild transformieren, durchs Netz, Ausgabe zurücktransformieren, mitteln.

Zwei Details: bei 90°/270° müssen in `wh` Breite und Höhe getauscht werden.
Und `off` wird nur aus dem untransformierten Durchlauf gelesen — seine korrekte
Rücktransformation als Vektorfeld hätte zusätzliche Vorzeichenfälle
eingeführt, und der Nutzen der TTA liegt im Finden, nicht im Nachjustieren um
Bruchteile einer Zelle.

Verifikation, dass die Rücktransformation stimmt: `selftest_equivariance`
korreliert die zurückgedrehte Heatmap mit dem untransformierten Durchlauf. Ein
Vorzeichenfehler ergäbe Korrelation ≈ 0; gemessen sind es 0.35–0.47 (Faltungen
sind nicht exakt rotationsäquivariant, deshalb nicht 1.0).

---

## 9 Training: was in der Praxis wehtut

### 9.1 Die Daten sind synthetisch

Es gab **keine** annotierten Trainingsbilder — nur 9 ausgeschnittene
Rohrenden und 7 Hintergründe. Der Trainingssatz wurde deshalb generiert:
6000 Bilder, 136 384 Labels, per Copy-Paste mit Rotation, Ovalisierung,
Okklusion, Farbangleich und JPEG-Requantisierung
(`../docs/PIPELINE.md`).

Der kritische Punkt dabei ist **Shortcut Learning**: bliebe an den
eingeklebten Objekten eine sichtbare Naht, ein Helligkeitsversatz oder eine
abweichende Rauschstatistik, würde das Netz „finde die Einfügemarke" lernen
statt „finde ein Rohrende". Auf synthetischen Daten wäre es dann exzellent, auf
echten unbrauchbar.

Der **Sim-to-Real-Abstand** ist deshalb eine wichtige Diagnose. Gemessen für
PipeNet (150 gehaltene synthetische Bilder gegen die beiden realen):

| Split | Art | mAP@0.5 | mAP@0.5:0.95 | F1 |
|---|---|---|---|---|
| `sval` (Trainingsmaßstab) | synthetisch | 0.9724 | 0.6591 | 0.9494 |
| `val` | real | 0.9060 | 0.6140 | 0.8778 |
| `test` | real | 0.9335 | 0.4531 | 0.9167 |

Der Abstand beträgt also **4–7 Punkte mAP@0.5**. Diese Zahl ist in beide
Richtungen aussagekräftig:

* **Klein genug**, dass das Verfahren praktisch funktioniert.
* **Groß genug**, dass das Netz offensichtlich **nicht** nur Einfügeartefakte
  gelernt hat. Hätte es das, wäre `sval` nahezu perfekt (mAP > 0.99) und die
  realen Bilder katastrophal.

Interessanter Nebenbefund: bei den YOLO-Modellen beträgt derselbe Abstand
**9.6 Punkte** (0.982 synthetisch gegen 0.886 real). PipeNet generalisiert
also *besser* von synthetisch nach real — was zur Kapazitätsüberlegung aus
Abschnitt 5.1 passt: weniger Parameter, weniger Spielraum, sich auf
Renderartefakte festzulegen.

### 9.2 Augmentierung

Zur Trainingszeit kommen dazu: **Mosaic** (vier Bilder in eine Leinwand — mehr
Objekte je Iteration, Objekte an Kachelrändern), Skalierung 0.6–1.4,
Zufallsausschnitt, Spiegelung horizontal und vertikal, HSV-Jitter. Alles
identisch zum YOLO-Vergleichslauf, damit der Vergleich ein
Architekturvergleich bleibt.

**Keine** Bildrotation — die würde achsparallele Boxen mitdrehen und
aufblähen (10° ⇒ +17 % Kantenlänge bei quadratischer Box), also Labelrauschen
erzeugen. Rotationsinvarianz liefert stattdessen der Generator, der jede
Instanz einzeln dreht und die Box danach exakt aus der gedrehten Alphamaske
neu bestimmt.

### 9.3 Diskriminative Lernraten

Der Encoder ist vortrainiert, Dekoder und Köpfe sind zufällig initialisiert.
Eine einheitliche Lernrate würde die brauchbaren Encoder-Gewichte in den
ersten Epochen zerstören, während die Köpfe noch gar keinen sinnvollen
Gradienten liefern. Deshalb: Encoder mit `0.1 × lr`, Rest mit `lr`. Dazu
3 Aufwärm-Epochen mit linear ansteigender Lernrate und danach Kosinus-Abfall.

### 9.4 Ein Systems-Problem am Rande

In dieser Umgebung (opencv-python 5.0.0.93 + torch 2.13) verklemmt sich
**jeder** fork-basierte DataLoader beim ersten Batch, sobald `num_workers > 0`
— nachgewiesen mit dem Ultralytics-Loader und mit unserem eigenen Datensatz,
und unabhängig von `cv2.setNumThreads(0)`, deaktiviertem OpenCL, gesetzten
OMP-Limits und Startmethode `spawn`.

Statt auf `workers=0` zurückzufallen, lädt PipeNet mit einem eigenen
**Thread-Pool** (`src/loader.py`). Threads forken nicht, und sie helfen hier
wirklich: die teuren Stufen (`cv2.resize`, `cv2.cvtColor`, `cv2.LUT`, große
numpy-Kopien) geben die **GIL** frei und laufen echt parallel.

| Variante | ms/Sample | min/Epoche |
|---|---|---|
| seriell | 31.9 | 3.19 |
| 4 Threads | 5.0 | 0.50 |
| 12 Threads | 2.5 | 0.25 |

Faktor 12.8. Dazu Normalisierung auf der GPU statt auf der CPU (24 ms → ~0)
und Zuschneiden **vor** dem Skalieren statt danach. Insgesamt von 7.2 auf
0.75 min/Epoche.

Das ist kein Deep-Learning-Thema, sondern ein Nebenläufigkeits-Thema — und
genau die Art Engpass, die man beim Trainieren am häufigsten wirklich hat.

---

## 10 Evaluation: was mAP eigentlich misst

Eine Detektion gilt als **True Positive**, wenn sie eine Ground-Truth-Box mit
IoU ≥ τ trifft und diese GT-Box noch nicht von einer besser bewerteten
Detektion belegt ist. Sonst **False Positive**. Unbelegte GT-Boxen sind
**False Negatives**.

Fährt man die Konfidenzschwelle von hoch nach niedrig, erhält man eine
**Precision-Recall-Kurve**:

```
Precision = TP / (TP + FP)      „von dem, was ich melde, ist wie viel echt?"
Recall    = TP / (TP + FN)      „von dem, was da ist, finde ich wie viel?"
```

**AP** (Average Precision) ist die Fläche unter dieser Kurve, in der
COCO-Konvention über 101 Recall-Punkte interpoliert. **mAP@0.5** ist AP bei
τ = 0.5; **mAP@0.5:0.95** ist das Mittel über τ = 0.50, 0.55, …, 0.95.

### Warum mAP@0.75 bei 20-px-Objekten einbricht

Das ist der wichtigste Punkt für die Interpretation der Ergebnisse. Mit der
IoU-Formel von oben lässt sich ausrechnen, wie genau eine Box sitzen muss, um
eine gegebene Schwelle τ zu erreichen:

| τ | erlaubter Versatz Δ/d | bei d = 20 px | bei d = 27 px |
|---|---|---|---|
| 0.50 | 0.184 | 3.7 px | 5.0 px |
| 0.75 | 0.074 | **1.5 px** | 2.0 px |
| 0.90 | 0.027 | 0.5 px | 0.7 px |

Bei einem 20-px-Objekt darf die Box für τ = 0.75 also um höchstens **1.5
Pixel** falsch sitzen — für τ = 0.5 sind noch 3.7 px erlaubt. Genau daraus
entsteht die Schere zwischen den beiden Metriken.

Deshalb ist mAP@0.75 auf `test` (Median 19.6 px) bei allen Modellen um 0.33
bis 0.39, auf `val` (Median 27.4 px) aber bei 0.73 bis 0.84 — bei fast
identischem mAP@0.5. **Der Unterschied ist Pixelauflösung, nicht
Detektionsqualität.** Wer mAP@0.5:0.95 als „die" Metrik nimmt, misst bei
Kleinstobjekten überwiegend, wie groß die Objekte im Bild sind.

Für die Anwendung (Zählen) ist ohnehin nicht mAP entscheidend, sondern der
**Zählfehler** an einem festen Betriebspunkt.

### Ergebnis

Auf dem Testbild, das nie im Training oder in der Modellauswahl vorkam:

| Modell | Par. | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | F1 | IoU (TP) | Zählung |
|---|---|---|---|---|---|---|---|
| `yolo11s` | 9.43 M | 0.9279 | 0.3398 | 0.4482 | 0.9167 | 0.745 | 82/86 |
| `yolo26s` | 9.95 M | 0.9051 | 0.3326 | 0.4270 | 0.8876 | 0.729 | 83/86 |
| `yolo26s-p2` | 9.66 M | **0.9471** | 0.3257 | 0.4492 | **0.9341** | 0.739 | 81/86 |
| **PipeNet** | **3.07 M** | 0.9335 | 0.3548 | 0.4531 | 0.9167 | 0.742 | 82/86 |
| **PipeNet +TTA** | **3.07 M** | 0.9220 | **0.3775** | **0.4648** | 0.9102 | **0.756** | 81/86 |

**Wichtige Einordnung:** das Bootstrap-Konfidenzintervall für mAP@0.5 liegt bei
±0.04–0.05, weil je Split nur **ein** annotiertes Bild existiert. **Alle**
Unterschiede in dieser Tabelle liegen darin. Interpretierbar ist nicht die
einzelne Differenz, sondern das Muster.

---

## 11 Was der Entwurf gebracht hat — und was nicht

### Bestätigt

**Stride 4 + Subpixel-Kopf verbessern die Lokalisierung.** PipeNet erreicht die
besten Werte aller fünf Modelle bei mAP@0.75, mAP@0.5:0.95 und mittlerer IoU
— mit einem Drittel der Parameter. Besonders aufschlussreich: `yolo26s-p2` hat
ebenfalls eine P2-Ebene, erreicht aber das **niedrigste** mAP@0.75. Der
Unterschied ist der Versatzkopf; YOLO26 hat `reg_max = 1` und damit keinen
vergleichbaren Subpixel-Mechanismus. **Feine Auflösung allein genügt nicht.**

**Peak-Extraktion funktioniert ohne Schwellenparameter.** Kein Fall
verschmolzener Zentren, und der komplette NMS-IoU-Sweep entfällt, weil der
Parameter nicht existiert.

**Kleine Kapazität ist hier kein Nachteil.** 3.07 M gegen 9.4–10.0 M, und bei
drei von sechs Metrikspalten der beste Wert.

### Nicht bestätigt

**Die Skalierungsregel.** Im YOLO-Teil hatte ich formuliert: so skalieren, dass
die Objektgröße dem Trainingsmedian (22.5 px) entspricht. Das Optimum liegt
aber bei 41–49 px. Die tragfähigere Erklärung ist die **Zellzahl**: bei
Stride 4 belegt ein 41-px-Objekt 10.3 statt 5.5 Zellen, was Peak und
Größenschätzung genauer macht. Nach oben begrenzt die augmentierte
Trainingsspanne (7–62 px).

**Die TTA hilft nicht überall.** Sie verbessert die Lokalisierung (mAP@0.75
+2.3, IoU +1.4 Punkte) und verschlechtert die Detektion (mAP@0.5 −1.2, F1
−0.7). Erklärung: gemittelt werden acht nicht deckungsgleiche Antworten; das
glättet Größe und Position, verbreitert aber die Maxima, sodass schwache Peaks
unter die Schwelle fallen.

### Wo PipeNet zurückliegt und warum

`yolo26s-p2` ist bei mAP@0.5 und F1 vorn. Zwei benennbare Gründe:

1. **Vortraining ist nicht kontrolliert.** Bei PipeNet sind nur die 2.78 M
   Encoder-Parameter ImageNet-vortrainiert; Dekoder und Köpfe (0.29 M) starten
   zufällig. YOLO bringt **vollständig COCO-vortrainierte Detektionsköpfe**
   mit. Das ist der klarste verbleibende Hebel und kein Architekturvorteil.
2. **Die Pyramide hilft an den Größenrändern.** Bei einer Ebene erreichen 12-px-
   und 60-px-Objekte 3 bzw. 15 Zellen. Das erklärt den Recall-Unterschied
   (0.907 gegen 0.895).

### Ein Befund über beide Architekturen hinweg

Beide PipeNet-Läufe erreichen ihr bestes Modell in **Epoche 6** — auch der mit
vollständig auslaufender Kosinusrampe. Bei YOLO: Epoche 10, 5 und 22. Immer
weit vor dem Budgetende.

Die Grenze ist also nicht die Optimierung, sondern der **Sim-to-Real-Abstand**.
Nach wenigen Epochen hat das Netz alles gelernt, was die synthetischen Daten
über die reale Domäne aussagen; danach verbessert es nur noch seine Anpassung
an die synthetische Verteilung. Praktische Folge: **mehr Trainingsaufwand ist
der falsche Hebel, besseres Quellmaterial der richtige.**

---

## 12 Glossar

| Begriff | Bedeutung |
|---|---|
| **Faltung / Convolution** | Kleiner Kernel wird über das Bild geschoben; Gewichte werden gelernt statt vorgegeben. |
| **Kanal / Merkmalskarte** | Eine Ausgabeebene pro Kernel. Tensorform `(C, H, W)`. |
| **Stride** | Verkleinerungsfaktor gegenüber dem Eingang. Stride 4 = Karte ist ¼ so groß. |
| **P2, P3, P4, P5** | Pyramidenebenen mit Stride 4, 8, 16, 32. |
| **Rezeptives Feld** | Der Bildbereich, der eine Ausgabezelle beeinflusst. |
| **Residualblock** | `out = f(x) + x`. Die Abkürzung lässt den Gradienten ungehindert zurückfließen. |
| **FPN** | Feature Pyramid Network: Top-down-Pfad, der grobe Semantik mit feiner Auflösung verbindet. |
| **Kopf / Head** | Kleines Faltungspaar, das Merkmale auf die gewünschte Ausgabe abbildet. |
| **Heatmap** | Einkanalige Karte mit Zentrumswahrscheinlichkeiten. |
| **IoU** | Schnittfläche / Vereinigungsfläche zweier Boxen. |
| **NMS** | Non-Maximum Suppression: Duplikate über eine IoU-Schwelle entfernen. In PipeNet nicht nötig. |
| **Focal Loss** | Verlust, der leichte Beispiele dämpft; nötig bei extremem Klassenungleichgewicht. |
| **Logit** | Wert vor der Sigmoid-Funktion. `logit(p) = log(p/(1−p))`. |
| **AP / mAP** | Fläche unter der Precision-Recall-Kurve, bei einer bzw. gemittelt über mehrere IoU-Schwellen. |
| **TTA** | Test-Time-Augmentation: mehrfach transformiert vorhersagen und mitteln. |
| **D4** | Diedergruppe: die 8 Symmetrien des Quadrats (4 Drehungen × 2 Spiegelungen). |
| **Shortcut Learning** | Das Netz lernt ein zufällig korreliertes Merkmal statt der Aufgabe. |
| **Sim-to-Real-Abstand** | Leistungsabfall beim Wechsel von synthetischen auf echte Daten. |
| **GIL** | Global Interpreter Lock: erlaubt nur einem Python-Thread Bytecode-Ausführung. C-Bibliotheken wie OpenCV geben ihn frei. |

---

## Wo im Code

| Datei | Inhalt |
|---|---|
| `src/model.py` | Architektur, Encoder-Abgriffe, Dekoder, Köpfe |
| `src/dataset.py` | Zielkodierung (Abschnitt 6), Augmentierung |
| `src/losses.py` | Focal Loss und maskierter L1 (Abschnitt 7) |
| `src/decode.py` | Peak-Extraktion, D4-TTA, Overlay-Geometrie (Abschnitt 8) |
| `src/loader.py` | Thread-Pool-Datenlader (Abschnitt 9.4) |
| `src/train.py` | Trainingsschleife, alle Parameter mit Begründung |
| `src/eval_pipenet.py` | Metriken (Abschnitt 10), Skalierungs-Sweep |
| `src/compare.py` | Gegenüberstellung mit den YOLO-Modellen |

Ausprobieren:

```bash
.venv/bin/python cnn/src/model.py            # Parameterzahl, Ausgabeformen
.venv/bin/python cnn/src/dataset.py          # sigma-Verteilung der Zielglocken
.venv/bin/python cnn/src/loader.py           # Datenlader-Benchmark
.venv/bin/python cnn/src/predict.py BILD.jpg --heatmap
```

*Eine stark vereinfachte Fassung (Zielgruppe ~10 Jahre) liegt in
[`docs/einfach_kinderfassung.md`](docs/einfach_kinderfassung.md).*
