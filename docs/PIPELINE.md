# Data-Augmentation-Pipeline — Aufbau, Transformationen, Anteile

Erzeugt durch `src/matting.py` → `src/occluders.py` → `src/generate.py`.
Alle Anteile sind **gemessene Ist-Werte** aus `synth/train/generation_report.json`
(6000 Bilder, 173 100 gerenderte Instanzen, 136 384 gelabelte Objekte), nicht
nur konfigurierte Sollwerte.

---

## 0. Leitprinzip

Die Pipeline verfolgt genau ein Ziel: **jede Abkürzung schließen, über die ein
CNN „hier wurde etwas eingeklebt“ statt „hier ist ein Rohrende“ lernen könnte.**
Bei Copy-Paste-Synthese ist das die dominante Fehlerquelle — nicht zu wenig
Variation, sondern *systematische* Unterschiede zwischen eingefügten und
originalen Pixeln. Jede Transformation unten wird deshalb entweder mit einer
Messung an den Realdaten oder mit einer konkreten Abkürzung begründet, die sie
schließt.

Zweites Prinzip — **Arbeitsteilung**:

| macht der Generator (offline) | macht Ultralytics (zur Trainingszeit) |
|---|---|
| alles, was **pro Instanz** wirkt und die Box **label-exakt** neu berechnet: Rotation, Ovalisierung, Delle, Perspektive, Farbe, Okklusion | alles, was **bildweit** wirkt und Boxen nicht verfälscht: Mosaic, Skalierung, Translation, Spiegelung, HSV |

Deshalb steht im Training `degrees=0`, `shear=0`, `perspective=0`: eine
Bildrotation dreht achsparallele Boxen mit und bläht sie auf (10° ⇒ +17 %
Kantenlänge bei quadratischer Box) — das wäre Labelrauschen. Im Generator wird
die Box nach der Rotation aus der gedrehten Alphamaske **exakt neu bestimmt**.

---

## Stufe 1 — Matting (`src/matting.py`)

**Problem.** Alle 9 Cutouts sind RGBA mit durchgängig `alpha = 255`, also
rechteckige Ausschnitte. 13–35 % ihrer Fläche ist Fremdkontext. Direktes
Einkopieren erzeugt eine achsparallele Naht — die perfekte Abkürzung.

**Lösung, dreistufig:**

1. **Manuelle Startwerte** für die Ellipse (cx, cy, rx, ry, θ), abgelesen aus
   einer 14-fach vergrößerten Rasteransicht. Bei 9 Objekten ist das exakter als
   jeder automatische Fit.
2. **Numerische Verfeinerung** — Coordinate-Descent, maximiert die radiale
   Kantenenergie `I(innen) − I(außen)` entlang der Ellipse auf einem 6-fach
   upgesampelten, geglätteten Bild.
3. **Außenkanten-Korrektur.** Schritt 2 konvergiert konstruktionsbedingt auf den
   *Grat* des hellen Kranzes. Ein mittleres Radialprofil bestimmt zusätzlich den
   Punkt, an dem die Helligkeit auf 25 % der Kantenamplitude über
   Hintergrundniveau abgefallen ist. Warum 25 % und nicht der klassische
   50 %-Punkt (FWHM): der Kranz ist das **diskriminative Merkmal**; ihn
   anzuschneiden wäre teurer als ein ~1 px breiter Hintergrundsaum, den
   Feathering und Farbangleichung später ohnehin kaschieren. Aufweitung hart auf
   [1.02, 1.10] gedeckelt — bei acht Cutouts liegt der Wert bei 1.02–1.05, nur
   das kontrastarme `cutted_8.png` liefe sonst auf 1.19 und die Maske würde
   Hintergrund aufsaugen.

**Ergebnis:** weiche Alphamaske mit 1.1 px Übergangsbreite. Kontrolle:
`docs/matting_check.png` (links Ellipse über Original, rechts freigestellt auf
neutralem Grau).

Der Übergang ist bewusst weich: ein 25-px-Objekt hat in einer echten Aufnahme
nie eine pixelharte Silhouette.

---

## Stufe 2 — Okkluder-Bibliothek (`src/occluders.py`)

**Warum echte Stangen statt gezeichneter Balken.** Ein synthetisch gemalter
Balken hätte andere Textur-, Rausch- und Glanzstatistik als der Rest des Bildes.
Das Netz könnte „Objekt liegt neben künstlichem Balken“ lernen. Die vier
Clean Plates sind buchstäblich Wände aus Rohrschäften desselben Materials,
derselben Kamera, derselben Beleuchtung.

**Verfahren:**
1. Dominante Rohrrichtung über den **Struktur-Tensor**, Winkel auf (−45°, 45°]
   gefaltet, Bild entsprechend gedreht (gemessen: +0.9°, −1.6°, +0.9°, +6.6°).
2. Zeilenmittelprofil, Hochpass, lokale Maxima = **Rohrachsen**; ihr Abstand =
   Durchmesser (gemessen: 16, 18, 21, 20 px).
3. Streifen der Höhe = Achsabstand um jede Achse ausschneiden.
4. Weiche **Kapselmaske** (Rechteck mit halbrunden Enden, 1.6 px Feder).

**Ergebnis:** 56 Stangen mit echter zylindrischer Schattierung und echtem
Glanzlicht (`docs/bars_check.png`).

---

## Stufe 3 — Bildsynthese (`src/generate.py`)

Leinwand **1024×1024**, identisch zur Trainings-`imgsz` — die synthetischen
Bilder durchlaufen im Training **keine** Skalierung, die kalibrierte
Objektgrößenverteilung bleibt exakt erhalten.

### T1 — Hintergrund

| Transformation | Anteil (Ist) | Begründung |
|---|---|---|
| Clean Plate statt Fremdszene | 80 % konfiguriert (6-faches Ziehgewicht) | Die vier Clean Plates sind exakt die Zieldomäne (Rohrstapel). Die drei Fremdszenen liefern zu 20 % Kontextvielfalt und harte Negative (`bg_5.jpg` enthält Maschendraht — die Struktur, die als Falschpositiv droht). |
| Zoom **gekoppelt an den Rohrdurchmesser** | 100 % | **Physikalische Randbedingung, Verhältnis gesetzt.** Stirnflächen und Rohrschäfte derselben Aufnahme werden annähernd im selben Maßstab abgebildet. Deshalb wird nicht Zoom und Objektgröße unabhängig gezogen, sondern `d = pipe_d(bg) × zoom`. Zufällige Missverhältnisse würden eine falsche Kontext-Größen-Relation antrainieren. Gesetzt ist das Durchmesserverhältnis 1:1: Die seitlichen Rahmenrohre, deren Stirnflächen detektiert werden, müssen nicht denselben Durchmesser haben wie die Rohre der Clean Plates. |
| Zufallsausschnitt (= **Translation**) | 100 % | Verhindert, dass immer derselbe Bildbereich als Hintergrund dient. |
| Reflect-Padding statt Kachelung | nur bei Bedarf, max. 1.35 × kürzeste Kante | `np.tile` erzeugt harte achsparallele Nähte — genau die Art Artefakt, die ein CNN als Merkmal missbraucht. Gespiegeltes Padding hat keine Kante. |
| Horizontale Spiegelung | **50.5 %** | Label-exakt, physikalisch korrekt: ein Stapel wird von beiden Enden fotografiert. |
| Vertikale Spiegelung | **14.9 %** | Ebenfalls label-exakt. Bewusst seltener als horizontal, weil die Schwerkraft (Stapel unten, Himmel oben) doch eine schwache Vorzugsrichtung erzeugt. |
| Kleinrotation ±6° | **34.3 %** | Die Rohrlinien sind nie exakt achsparallel. Ohne diese Variation wäre die Vorzugsrichtung im Datensatz künstlich diskret. |

### T2 — Instanz-Geometrie

| Transformation | Anteil (Ist) | Begründung |
|---|---|---|
| **Rotation 0–360°** | **100 %** | Rohrenden haben keine kanonische Bildorientierung: Kamerarollwinkel und Stapelausrichtung sind beliebig. Die einzige Orientierungsinformation ist die Richtung des Glanzlichts auf dem Kranz, und die hängt vom Sonnenstand ab. Volle Rotationsinvarianz ist daher korrekt **und kostenlos** — es gibt keine Rotation, die ein unmögliches Bild erzeugt. Deshalb 100 % statt eines Bruchteils. |
| **Skalierung** auf 12–44 px | **100 %** | Reale Messung: val 20.6–35.8 px (Median 27.4), test 12.2–31.9 px (Median 19.6). Der Generator deckt 12–44 px ab, also die reale Spanne plus ~25 % Reserve nach oben für Aufnahmen aus geringerem Abstand. Ist-Verteilung: p05 11.6 / Median 22.5 / p95 39.9 px. Verkleinerung mit `INTER_AREA`, weil `INTER_LINEAR` Aliasing als verräterisches Hochfrequenzmuster im Objekt zurückließe. |
| **Ovalisierung** (anisotrope Skalierung) | **57.1 %** (45 % oblat 0.66–0.97, 12 % prolat 1.03–1.30) | Reale Messung: Seitenverhältnis w/h val p10–p90 = 0.97–1.31, test 0.85–1.22 — also weicht **über die Hälfte** der echten Boxen um ≥ 5 % vom Quadrat ab. Zwei physikalische Ursachen: (a) Schrägansicht projiziert den Kreis auf eine Ellipse, (b) tatsächlich gequetschte Rohre. Der Anteil 57 % bildet die gemessene Häufigkeit direkt ab. Oblat dominiert (45:12), weil Schrägansicht die Höhe verkürzt und dieser Fall häufiger ist als eine seitliche Quetschung. |
| **Delle** (radiale Sektor-Einbeulung) | **18.1 %** | Modelliert mechanisch beschädigte Rohrenden. Umgesetzt als glatte radiale Verschiebung über 1–2 Winkelsektoren (Gauß-Fenster, Eindringtiefe 10–30 %, quadratisch mit dem Radius gewichtet, damit das Zentrum ruhig bleibt). **Der Anteil ist eine Annahme, keine Messung** — in val/test sind Dellen nicht separat annotiert, sodass ihre Häufigkeit unbekannt ist. 18 % wurde gewählt, damit das Modell die Verformung regelmäßig sieht (etwa jedes 5. Objekt, im Mittel 4 pro Bild), sie aber die Formstatistik nicht dominiert. Ein zu hoher Anteil würde die Rundheit als Merkmal verwässern und die Falschpositivrate auf beliebigen dunklen Flecken erhöhen. |
| **Perspektive** (milde Homographie ±9 %) | **30.2 %** | Bei Schrägblick wird der Kreisring nicht nur zur Ellipse — der helle Kranz wird auf einer Seite auch **breiter**, weil man die Rohrinnenwand sieht. Eine reine anisotrope Skalierung kann das nicht abbilden. 30 %, weil in val/test die Mehrheit der Stirnflächen annähernd frontal steht (Fotograf steht vor dem Stapelende) und ausgeprägte Schrägansicht die Ausnahme ist. |

### T3 — Instanz-Photometrie und Einbettungsrealismus

| Transformation | Anteil (Ist) | Begründung |
|---|---|---|
| Helligkeit ×0.70–1.35 | **99.8 %** | Stirnflächen liegen je nach Stapelposition in praller Sonne oder tiefem Schatten. Die Spannweite ist direkt aus den Cutouts ablesbar: mittlere Kranzhelligkeit 88–156 Grauwerte. Praktisch 100 %, weil es keine „richtige“ Helligkeit gibt, die bevorzugt werden dürfte. |
| **Gamma** 0.75–1.35 | **79.7 %** | Gamma statt reinem Kontrast, weil es die nichtlineare Tonwertkurve verschiedener Kameras und JPEG-Pipelines abbildet. Es verändert vor allem das Verhältnis **dunkle Bohrung ↔ heller Kranz** — genau das Merkmal, auf das der Detektor sonst zu starr konditioniert würde. |
| HSV (Farbton ±8°, Sättigung ×0.45–1.60) | **69.7 %** | Verzinkter Stahl ist nahezu neutralgrau, nimmt aber Farbstiche an: Flugrost braun, Moos grün, blauer Himmel als Reflex. Farbtonhub bewusst klein — große Sprünge wären unphysikalisch. |
| **Farbangleich an den lokalen Hintergrund** | **84.8 %** | **Die wichtigste Anti-Abkürzungs-Maßnahme.** Cutout und Hintergrund stammen aus verschiedenen Aufnahmen mit verschiedenem Weißabgleich und Belichtungsniveau; ohne Angleich bleibt ein konstanter Farb-/Helligkeitsversatz, den ein CNN trivial als Einfügemarke lernt. Bewusst **nur teilweise**: Mittelwertangleich α ≤ 0.45, Streuungsangleich ≤ 0.35, Streuungsverhältnis auf [0.6, 1.6] geklemmt. Vollständiger Angleich würde den intrinsischen Albedo-Unterschied Bohrung/Kranz einebnen und das Objekt zerstören. Modelliert wird also die *Beleuchtungs-/Weißabgleichdifferenz*, nicht die Objektfarbe. |
| **Unschärfeangleich** (σ 0.25–1.10 px) | **59.8 %** | Zwei Gründe: (a) das skalierte Cutout ist je nach Skalierungsrichtung glatter oder schärfer als das Zielbild, (b) reale Objekte liegen je nach Stapeltiefe außerhalb der Schärfeebene. Schärfe ist eine der beiden Sensorsignaturen, an denen eingeklebte Pixel am leichtesten erkennbar sind. |
| **Rauschangleich** (σ aus der lokalen Hochpass-Streuung geschätzt, ×0.5–1.25) | **69.9 %** | Die zweite Sensorsignatur. Beim Verkleinern eines Cutouts sinkt dessen Rauschen; der eingeklebte Bereich wäre dann messbar „sauberer“ als seine Umgebung. Das Rauschniveau wird **pro Einfügestelle gemessen**, nicht global gesetzt (Hintergründe: 6.8–15.2 Grauwerte Hochpass-σ). |
| Alphakante zusätzlich weichzeichnen (σ 0.4–1.0 px) | **100 %** | Immer. Eine pixelharte Silhouette gibt es bei 25-px-Objekten in echten Aufnahmen nicht. |
| **Schlagschatten** | **49.3 %** | Ohne Kontaktschatten „schwebt“ ein eingeklebtes Objekt sichtbar — es ist derselbe Hinweis, den auch ein menschlicher Betrachter zum Erkennen einer Montage nutzt. ~50 %, weil bei diffusem Licht (alle vier Clean Plates sind bedeckt aufgenommen) etwa die Hälfte der realen Objekte keinen sichtbaren Schlagschatten wirft. |

### T4 — Layout (datengetrieben)

Modell: **gescherte, verrauschte Gitterpackung** je Cluster; 1–3 Cluster pro Bild
(45/35/20 %), dazu 0–3 vereinzelte Profile.

Kalibriert an der Realmessung (n = 174):

| Kenngröße | real val | real test | **synthetisch (Ist)** |
|---|---|---|---|
| NN-Abstand p10 / Median / p90 (in Ø) | 0.86 / 1.09 / 1.78 | 0.83 / 1.24 / 2.51 | **0.84 / 1.17 / 1.87** |
| Objekte, die ihren Nachbarn berühren (< 1.2 Ø) | 61.4 % | 46.5 % | **53.6 %** |
| Nachbarn innerhalb 5 Ø (Median) | 10 | 10 | **13** |
| Größenverhältnis zum Nachbarn (Median) | 1.05 | 1.05 | **1.05** (per Konstruktion) |

Zwei Modellentscheidungen folgen direkt aus der Messung:

* **Eine Basisgröße pro Cluster**, je Instanz nur ±8 % Streuung. Innerhalb eines
  Stapels haben alle Rohre dieselbe Nennweite — das gemessene Größenverhältnis
  benachbarter Objekte liegt im Median bei 1.05. Jede Instanz unabhängig zu
  würfeln wäre ein sofort sichtbarer Realismusfehler.
* **Cluster-Zentrum darf außerhalb der Leinwand liegen** → angeschnittene Objekte
  am Bildrand (in test sind 5.8 % der Labels angeschnitten).

Zusätzlich: **abgedunkelte Mulde** unter dem Cluster in **87.4 %** der Bilder.
In den echten Aufnahmen liegen die Stirnflächen am Stapelrand vor dem
verschatteten Stapelinneren; der Hintergrund hinter ihnen ist deutlich dunkler
als die besonnten Schäfte davor.

### T5 — Rohrschaft hinter jedem Profil (physikalische Randbedingung)

| Transformation | Anteil (Ist) | Begründung |
|---|---|---|
| **Echter Rohrschaft hinter der Stirnfläche** | **88.0 %** der Instanzen | **Die wichtigste Einzelentscheidung der Pipeline.** Die Sichtung der Realbilder zeigt ausnahmslos: hinter jeder annotierten Stirnfläche führt ein sichtbarer Rohrschaft weg. Ein dunkler Kreis mitten auf einem durchgehenden Rohr ist physikalisch unmöglich — die erste Generatorversion erzeugte genau das und sah entsprechend falsch aus. Der Schaft hat **exakt** den Durchmesser der Stirnfläche (dasselbe Rohr, ±5 %), Länge 2–24 Ø. Nebeneffekt: Schäfte erzeugen genau die gegenseitige Verdeckung, die in den echten Bildern dominiert. |
| Schaftrichtung je Cluster parallel (±11°) | 100 % | Alle Rohre eines Stapels liegen parallel; sie verkanten nur leicht. ±18° erzeugte sichtbar einen sternförmigen Fächer, den es real nicht gibt. |
| Schaft-Helligkeitsangleich an die Umgebung | 100 % (k = 0.35–0.75) | Die Stangenbibliothek stammt aus vier Aufnahmen mit stark unterschiedlichem Belichtungsniveau (Hintergrundmittel 70–102 Grauwerte). Ohne Angleich klebten sichtbar zu dunkle Rohre im Bild. |
| **75 % der Schäfte ganz hinten, 25 % eingereiht** | — | **Tiefenmodell.** Die Stirnflächen bilden eine Front; die Schäfte laufen von dieser Front weg, liegen in der Projektion also *hinter* allen Stirnflächen. Nur quer vor dem Stapel liegende Rohre kreuzen vor anderen Enden (in val/test sichtbar, aber die Minderheit). **Messbarer Effekt:** mit rein zufälliger Tiefenordnung verdeckten die Schäfte so viele Nachbarn, dass deren Labels wegfielen und der synthetische NN-Abstand auf 1.55 Ø stieg — 40 % über dem realen Wert. Mit dem 75/25-Modell: 1.17 Ø. |

### T6 — Okklusion

Der Aufgabenstellung nach ausdrücklich gefordert; sie entsteht hier aus **vier**
verschiedenen, jeweils physikalisch motivierten Quellen:

| Quelle | Anteil (Ist) | Begründung |
|---|---|---|
| **Andere Profile** (Überlappung durch Tiefenordnung) | intrinsisch | Bei einem NN-Abstand von 0.84 Ø im 10. Perzentil überlappen benachbarte Stirnflächen zwangsläufig. Die Tiefenordnung ist zufällig, das Verdeckungsverhältnis damit unverzerrt. |
| **Echte Rohrstangen quer davor** | **58.4 %** der Bilder, 1–6 Stangen | Rohre, die vor dem Stapel liegen. Breite 0.55–1.35 Ø, Länge 4–26 × Breite, bevorzugt (80 %) durch ein Cluster gelegt statt ins Leere. |
| **Eigene Schäfte** (die eingereihten 25 %) | s. T5 | Physikalisch der häufigste Verdeckungsfall in den Realbildern. |
| **Fremdkörper** (Laub, Draht, Schnee) | **31.6 %** der Bilder, 1–4 Stück | Aus den Hintergründen selbst geschnitten, mit unregelmäßiger weicher Maske aus geglättetem Rauschen. In `test4_cutted.png` liegt reichlich Laub auf dem Stapel. |
| **Bildrand** | ~3 % der Objekte | siehe T4. |

**Ergebnis:** **30.1 %** aller gelabelten Objekte sind teilverdeckt (Sichtbarkeit
< 95 %), mittlere Sichtbarkeit der gelabelten Objekte **90.0 %**.

### T7 — Sichtbarkeitsrechnung und Labelerzeugung

Kein Copy-Paste-Generator darf Labels blind vergeben. Hier wird für **jede**
Instanz die exakte Sichtbarkeit gerechnet:

```
sichtbar_i = alpha_i · Π  (1 − alpha_j)      für alle j, die NACH i gemalt wurden
```

(inklusive Stangen und Fremdkörpern; die Buchhaltung läuft über die
Alpha-Fenster in Malreihenfolge).

Regeln:

| Regel | Wert | Begründung |
|---|---|---|
| **Label verwerfen** wenn Sichtbarkeit < 30 % | 30 801 Objekte (17.8 % der gerenderten) | Unter 30 % sichtbarer Fläche wäre das Objekt auch für einen menschlichen Annotator nicht mehr sicher als Rohrende identifizierbar. Ein Label hier zwingt das Netz zum Halluzinieren; kein Label macht es zum korrekten Hard Negative — dieselbe Regel, nach der val/test annotiert wurden. |
| **Label verwerfen** wenn < 15 % der Fläche im Bild liegt | 3 846 Objekte | Ein Splitter am Bildrand wird real nicht annotiert. 15 % und nicht mehr, weil `test` sehr wohl stark angeschnittene Objekte enthält (kleinste Box: 0.7 % Bildhöhe). |
| **Label verwerfen** wenn sichtbare Box < 5 px | 228 Objekte | Unter der Auflösungsgrenze. |
| **Modale Box** (sichtbarer Umriss), nicht amodale | — | Genau so wurden val und test annotiert: die Boxen umschließen den sichtbaren Teil, nicht den geschätzten vollständigen Kreis. Amodale Boxen würden einen systematischen Größenversatz gegenüber der Ground Truth erzeugen. |

### T8 — Globale Bildstufe

| Transformation | Anteil (Ist) | Begründung |
|---|---|---|
| Helligkeit/Kontrast | **89.6 %** | Wetter- und Belichtungsvariation. |
| HSV / Farbtemperatur | **69.0 %** | Bewölkt (blaustichig) bis Abendsonne (warm). Alle vier Clean Plates sind bei diffusem Licht aufgenommen — diese Stufe ist der einzige Ersatz für die fehlende Beleuchtungsvielfalt. |
| Defokus (σ 0.3–1.0 px) | **34.4 %** | Autofokus trifft bei Stapelaufnahmen oft die falsche Ebene. |
| **Bewegungsunschärfe** | **9.9 %** | Verwacklung aus der Hand — im Bestand plausibel, weil Baustellenaufnahmen im Vorbeigehen entstehen. Niedriger Anteil, weil moderne Handys kurz belichten. |
| Sensorrauschen | **80.1 %** | Vereinheitlicht das Rauschniveau über das ganze Bild. |
| Vignettierung | **25.0 %** | Weitwinkelobjektive von Mobiltelefonen; erzwingt Positionsinvarianz der Helligkeit. |
| **JPEG-Requantisierung q 55–95** | 85 % konfiguriert | **Die wichtigste „unsichtbare“ Transformation.** Eingeklebte Pixel behalten sonst eine andere DCT-Blockstatistik als der Hintergrund, der bereits einmal JPEG-komprimiert war. Ein erneuter Kompressionsdurchlauf über das **Gesamtbild** vereinheitlicht diese Signatur. Das ist forensisch die belastbarste Einfügemarke überhaupt — wer sie stehen lässt, liefert dem Netz eine perfekte Abkürzung. |

### T9 — Negativbilder

**9.7 %** der Bilder enthalten **kein** Objekt: Rohrstapel ohne Stirnflächen,
Kieswerk, Betondeck mit Maschendrahtzaun.

Begründung: die dominante Falschpositiv-Quelle in dieser Domäne sind
Rohrschäfte, Gitterknoten und dunkle Löcher im Maschendraht — alles Strukturen,
die den Stirnflächen ähneln. Explizite Negativbilder bestrafen genau diese
direkt. Der Wert ~10 % entspricht der Ultralytics-Empfehlung (0–10 %
Hintergrundbilder); mehr würde den Anteil positiver Gradienten unnötig
verdünnen.

---

## Zusammenfassung des erzeugten Datensatzes

| | |
|---|---|
| Bilder | 6 000 (1024×1024) |
| davon Negativbilder | 583 (9.7 %) |
| gerenderte Instanzen | 173 100 |
| **gelabelte Objekte** | **136 384** (22.7 pro Bild) |
| verworfen (Verdeckung/Rand/Größe) | 34 875 (20.1 %) |
| mittlere Sichtbarkeit gelabelter Objekte | 90.0 % |
| teilverdeckte gelabelte Objekte | 30.1 % |
| Objektdurchmesser p05 / Median / p95 | 11.6 / 22.5 / 39.9 px |
| Seitenverhältnis p10 / Median / p90 | 0.77 / 1.00 / 1.40 |

---

## Bekannte Grenzen

1. **Nur 9 Quellobjekte.** Jedes erscheint ~15 000-mal. Die photometrische und
   geometrische Streuung mildert das, hebt es aber nicht auf. Das ist der
   Hauptgrund, warum bewusst `yolo11s` und nicht `yolo11m/l` trainiert wird.
2. **Nur diffuses Licht in den Clean Plates.** Harte Schlagschatten, Gegenlicht,
   Nässe und Schnee fehlen im Quellmaterial vollständig und werden nur
   approximativ über die globale photometrische Stufe angenähert. Das ist die
   größte verbleibende Domänenlücke.
3. **Reflect-Padding erzeugt Spiegelsymmetrie**, wenn ein Hintergrund für den
   gewünschten Zoom zu klein ist (betrifft vor allem `test5_cutted.jpg`,
   945×552). Sichtbar in `docs/preview_train_004.jpg` am unteren Rand.
4. **Der Dellen-Anteil (18 %) ist eine Annahme**, keine Messung — siehe T2.
5. **Die Cluster sind im Mittel etwas dichter als real** (13 statt 10 Nachbarn
   innerhalb 5 Ø, Maximum 61 statt 20). Bewusst als Robustheitsreserve belassen.
