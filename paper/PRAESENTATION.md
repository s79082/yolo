# Vortragsplan — 20 Minuten

Grundlage: `kapitel_e2/*.tex`, Stand 31.08 — **der Zählbezug ist aus der Arbeit
entfernt, es geht ausschließlich um die Detektion der Profile.** Der Bau
(`make e2`) ergibt 41 Seiten Text, 62 Seiten gesamt, **0 offene Referenzen**.
Experiment 3 (P2-Kopf) steht in `07_auswertung.tex:241–522` im `\iffalse`-Block;
einziges Experiment ist die Okklusionsablation.

Alle Zahlen unten sind gegen die Quelldateien geprüft (und die des
Auflösungssweeps zusätzlich gegen die Rohdaten in
`docs/experiments/_yolo11s_backup/experiment1_inference.json` — sie stimmen).

---

## Der rote Faden in einem Satz

> Für diese Detektionsaufgabe gibt es keine Trainingsdaten — also erzeugt die
> Arbeit sie. Und weil dann die Pipeline *der Datensatz* ist, muss jeder ihrer
> 40 Parameter begründet sein statt nur eingestellt.

Alles andere hängt daran: Kapitel 3 liefert die Begründungspflicht und die
Anforderungen, Kapitel 5 erfüllt sie, Kapitel 7 prüft, ob das getragen hat.

---

## Zeitbudget

| # | Block | Kapitel | Zeit | kumuliert |
|---|---|---|---|---|
| 1 | Problem und Datenlage | 1 | 2:00 | 2:00 |
| 2 | Was Augmentierung formal leistet | 3 | 2:00 | 4:00 |
| 3 | Drei Besonderheiten bei der Detektion | 3 | 2:30 | 6:30 |
| 4 | Shortcut Learning, S1–S6, A1–A6 | 3 | 2:00 | 8:30 |
| 5 | Datengrundlage | 4 | 1:00 | 9:30 |
| 6 | Die Synthese-Pipeline | 5 | 5:00 | 14:30 |
| 7 | Trainingsaufbau | 6 | 0:30 | 15:00 |
| 8 | Ergebnisse | 7 | 3:40 | 18:40 |
| 9 | Bilanz und Grenzen | 8 | 1:00 | 19:40 |

Gegenüber der Zählfassung sind **50 Sekunden frei geworden** — der Absatz zur
Dichteregression in Block 1 (−30 s) und die F₁-gegen-Zählfehler-Herleitung in
Block 8 (−20 s) entfallen ersatzlos. 30 s davon gehen an Kapitel 5, das sie
besser gebraucht; 20 s bleiben Puffer.

**Kapitel 2 bekommt keinen eigenen Block.** Gebraucht wird daraus genau ein
Satz, und der passt in Block 3: Eine Zelle der feinsten Detektionsebene P3 deckt
8 × 8 Pixel ab, und weil der Kopf ankerfrei ist, entscheidet die Zahl der
Gitterzellen — nicht die Zahl der Ankerboxen —, wie fein Objekte überhaupt
aufgelöst werden. CNN-Grundlagen in einer Verteidigung durchzugehen kostet drei
Minuten und bringt keinen Punkt; die Zeit gehört Kapitel 5 und 7.

---

## Block 1 — Problem und Datenlage (2:30)

**Folie:** die reale Stapelaufnahme (`abb:realstapel`), daneben die drei Zahlen
**9 / 7 / 2**.

Mobile Bauzaunfelder werden liegend gestapelt, 50 bis 200 pro Stapel; für jede
automatisierte Erfassung eines Stapels müssen seine Einzelfelder im Bild
lokalisiert werden. Frontal aufgenommen zeigt jedes Feld die Stirnfläche seiner
Rahmenrohre: ein heller verzinkter Kranz mit dunkler Bohrung.

**Diesen Satz wörtlich sagen, er zieht die Scope-Grenze, bevor jemand danach
fragt:** Die Aufgabe dieser Arbeit ist eine *reine Detektionsaufgabe* — jede
sichtbare Stirnfläche ist mit einer Box zu lokalisieren. Sie ist die Vorstufe
jeder weiteren Auswertung eines Stapels; was danach mit den Boxen geschieht,
liegt außerhalb des Scopes und wird auch nicht bewertet.

Aus Sicht der Bildverarbeitung ist das ein unangenehmer Spezialfall: die Objekte
sind **12–36 px** groß, liegen im Median rund einen Objektdurchmesser
auseinander, sind untereinander nahezu identisch und wechselseitig verdeckt. Und
die Objektzahl verschärft jeden Fehler: Bei 50 bis 200 Stirnflächen pro Aufnahme
hinterlässt selbst ein Rückruf von **99 %** im Mittel noch ein bis zwei
übersehene Objekte je Bild.

Der Engpass ist also die Datenlage: COCO kennt keine Klasse „Bauzaunprofil", und
50–100 Boxen pro Bild über hunderte Bilder von Hand sind in einer Abschlussarbeit
nicht leistbar. Verfügbar sind **neun Objektausschnitte, sieben
Hintergrundaufnahmen und zwei annotierte Realbilder**.

Daraus die Entscheidung: **ausschließlich synthetisch trainieren, real
validieren und testen.** Damit ist die Augmentierungspipeline nicht mehr eine
Menge von Zusatztransformationen auf einem Datensatz — sie *ist* der Datensatz.

---

## Block 2 — Kapitel 3, Teil 1: Was Augmentierung formal leistet (2:00)

**Folie:** die Gleichung mit vier beschrifteten Bestandteilen, darunter eine
Zeile: *n = 9 + 7 · ein t = ein Generatordurchlauf · 6000 Bilder = 6000
Ziehungen aus p(t)*.

Statt des empirischen Risikos minimiert das Training das **augmentierte Risiko**

    R̂_aug(h) = (1/n) Σᵢ E_{t∼p(t)} [ L( h(t(xᵢ)), τ_t(yᵢ) ) ]

— die Vicinal Risk Minimization von Chapelle. Der Unterschied liegt allein im
Erwartungswert: Statt jedes Beispiel genau in der Form zu sehen, in der es
aufgenommen wurde, sieht das Netz eine ganze Umgebung davon.

Vier Bestandteile, und drei davon sind bei der Detektion nicht selbsterklärend:
**𝒯** legt fest, *welche* Operationen erlaubt sind; **p(t)**, *wie oft* und *wie
stark*; **τ_t** ist die Labeltransformation — wie y verändert werden muss, damit
es t(x) noch korrekt beschreibt. Bei der Klassifikation ist τ_t die Identität,
eine gespiegelte Katze bleibt eine Katze. Bei der Detektion nicht: Boxkoordinaten
müssen mit. Ist τ_t nur *näherungsweise* bekannt, entsteht Rauschen im Zielsignal.

**Die Pointe für diese Arbeit:** n ist winzig — als Ausgangsmenge lassen sich die
neun Ausschnitte und sieben Hintergründe lesen —, *ein* t ist der gesamte
Generatordurchlauf, und die 6000 Trainingsbilder sind 6000 Ziehungen aus p(t).

> Die gesamte **Vielfalt** des Datensatzes steckt damit in p(t) und in keinem
> einzigen aufgenommenen Bild. Deshalb *ist* die Dokumentation von p(t) die
> Dokumentation des Datensatzes.

(Sagen Sie „Vielfalt", nicht „Qualität". Die Qualität hängt zusätzlich an der
Exaktheit von τ_t — das ist genau der Unterschied, den A1 adressiert, und ein
Prüfer, der zuhört, hakt sonst ein.)

**Drei Wirkmechanismen**, knapp: *Kodierung von Invarianzen* — bildet 𝒯 eine
Gruppe, reduziert die Orbit-Mittelung die Varianz, ohne den Bias zu erhöhen,
Augmentierung ist also strikt vorteilhaft, **sofern die angenommene Invarianz
gilt**. *Regularisierung durch Rauschen* — Training mit Rauschen ist äquivalent
zu einer Tikhonov-Regularisierung. *Erweiterung des Trägers* — bei nur neun
Quellobjekten der dominierende Effekt.

Daraus der Entwurfsgrundsatz, der später A3 trägt: **Antrainiert werden dürfen
nur Invarianzen, die in der Domäne tatsächlich gelten.** Eine erzwungene, nicht
vorhandene Invarianz *verschlechtert* das Modell, weil sie diskriminative
Information entfernt. Volle Rotationsinvarianz ist bei einem Rohrende korrekt —
Kamerarollwinkel und Stapelausrichtung sind beliebig; bei einem Verkehrszeichen
wäre sie falsch.

---

## Block 3 — Kapitel 3, Teil 2: Drei Besonderheiten bei der Detektion (2:30)

**Gliederungssatz zuerst — das ist die Klammer des ganzen Abschnitts:**

> Die Box ist ein achsparalleles Rechteck und damit ein grobes Modell des
> Objektumrisses — bei 20 px großen, runden Objekten ein fragiles. Daraus folgt
> am **Eingang**, welche Transformation angewandt werden darf, ohne das
> Regressionsziel zu verfälschen; am **Ausgang**, welche Metrik gelesen werden
> darf, ohne einen Konventionsunterschied für einen Modellfehler zu halten.
> Beide laufen auf dieselbe Größe hinaus: das Verhältnis eines absoluten
> Boxfehlers zur Objektgröße. Die dritte Folge betrifft nicht die Box, sondern
> das Gitter.

### 1. Eingang: Rotation bläht Boxen auf

Bei einem rotationssymmetrischen Objekt — und genau das ist eine Rohrstirnfläche
— bleibt die *korrekte* Box unter Rotation gleich, die mitgedrehte achsparallele
Hülle wächst aber um |cos θ| + |sin θ|. Die Überlappung beider:

    IoU(θ) = 1 / (1 + |sin 2θ|)

Bei 2° noch 0,935, bei **10° schon 0,745**, bei 45° 0,500. Für ein 25-px-Objekt
heißt 10°: rund 4 px mehr Kantenlänge — systematisches Rauschen im Zielsignal der
Boxregression.

→ **Konsequenz: `degrees`, `shear` und `perspective` stehen zur Trainingszeit auf
0.** Nicht, weil die Invarianz unerwünscht wäre, sondern weil ein Generator sie
label-exakt erzeugen kann.

### 2. Ausgang: bei kleinen Objekten misst die IoU die Pixel

    IoU = (1 − δ/d) / (1 + δ/d)

Der Fehler wirkt **allein über das Verhältnis δ/d**. Ein Boxfehler von 2 px
kostet bei einem 20-px-Objekt 0,18 IoU, diagonal 0,32; bei einem 200-px-Objekt
wären es 0,02 und damit unsichtbar.

→ **mAP@0,5:0,95 ist bei dieser Objektgröße mehr ein Maß für die Pixelauflösung
als für die Detektionsqualität.** Das hier zu sagen ist wichtig — es ist die
Vorbereitung des stärksten Befunds in Block 8.

### 3. Gitter: unterhalb einer Grenze gibt es gar kein Label

Ein Detektor verarbeitet in fester Netzeingangsgröße M; ein Bild mit längster
Kante N wird um M/N skaliert:

    d_eff = d · M/N

Für N = 2459, d = 27 px und M = 640 kommt das Objekt mit **7,0 px** am Netz an —
weniger als die Schrittweite 8 der feinsten Standardebene. Und die erwartete Zahl
der Gitterzellen, deren Zentrum ins Objekt fällt und die damit bei einem
ankerfreien Kopf überhaupt als positive Zuweisung in Frage kommen, ist

    E[N_Zellen] = π d² / (4 S²)

Unterhalb von **d < S√2 ≈ 11,3 px** existieren Lagen, in denen *kein* Zellzentrum
im Objekt liegt — das Objekt bekommt dann **keine positive Zuweisung**. Genau im
Bereich der hier auftretenden Objektgrößen.

Die Literatur greift das an drei Stellen an: die Small-Target-Aware Label
Assignment von YOLO26 auf der **Verlustseite**, ein zusätzlicher P2-Kopf auf der
**Architekturseite**, Überabtastung kleiner Objekte auf der **Datenseite**.

→ Randbedingung für die Pipeline: Trainings- und effektive Inferenz-Größen­
verteilung müssen übereinstimmen. **Eine Trainingsskalierung, die Objekte unter S
schrumpfen lässt, erzeugt Labels ohne auflösbares Objekt.**

**Folien:** `abb:boxrotation` Teil (a) mit den drei IoU-Werten, und
`abb:aufloesung` Teil (b) — das Gitter mit und ohne Zellzentrum im Objekt.

---

## Block 4 — Kapitel 3, Teil 3: Warum synthetische Daten scheitern (2:00)

Der Nutzen hängt daran, wie ähnlich die erzeugte Verteilung D_S der
Zielverteilung D_T ist. Ben-David gibt dafür die Schranke

    ε_T(h) ≤ ε_S(h) + ½·d_HΔH(D_S, D_T) + λ

Sie listet die Angriffspunkte des Pipelineentwurfs vollständig auf: **(i)** ε_S
klein halten — korrekt gelabelte synthetische Daten, deshalb wird die
Sichtbarkeit exakt gerechnet; **(ii)** die Divergenz klein halten — Zweck aller
Kalibrierungen und aller Maßnahmen zum Einbettungsrealismus; **(iii)** λ klein
halten — die *Labeldefinition* muss in Quelle und Ziel identisch sein.

**Die entscheidende Struktur von (ii):** Die Divergenz wird über die
Hypothesenklasse gemessen. Ein Unterschied, den ein CNN *leicht* erkennen kann —
eine achsparallele Naht, ein konstanter Farbversatz — trägt weit stärker bei als
einer ohne einfache Entscheidungsfunktion.

Damit ist man beim **Shortcut Learning**: Netze nutzen bevorzugt die Merkmale,
die auf den Trainingsdaten am einfachsten zum Ziel führen. Bei eingeklebten
Objekten sind diese Abkürzungen benennbar — die Bildforensik untersucht genau
sie, um Montagen nachzuweisen. Sechs Signaturen, jede mit einer Gegenmaßnahme:

| | Signatur | Gegenmaßnahme |
|---|---|---|
| S1 | geometrische Naht | Freistellen mit weicher Alphamaske |
| S2 | Farb- und Helligkeitsversatz | partieller Statistikangleich |
| S3 | Schärfe- und Auflösungsversatz | flächentreue Verkleinerung, Unschärfeangleich |
| S4 | Rauschversatz | Rauschangleich mit lokal geschätzter Amplitude |
| S5 | Kompressionsversatz | erneute Requantisierung des *Gesamtbildes* |
| S6 | fehlende Szeneninteraktion | Schlagschatten, physikalisch motivierte Verdeckung |

**Der Satz, der die ganze Pipeline begründet:**

> Bei Copy-Paste-Synthese ist *nicht zu wenig Variation* die dominante
> Fehlerquelle, sondern der *systematische* Unterschied zwischen eingefügten und
> originalen Pixeln. Variation lässt sich durch mehr Zufall erzeugen; ein
> systematischer Versatz bleibt bei beliebig viel Zufall bestehen und wird vom
> Netz zuverlässig gefunden.

**Was nicht geht — und jeweils aus einem anderen Grund.** Diese Präzision lohnt
sich, weil genau danach gefragt wird: AutoAugment, RandAugment und
Detektions-Policies brauchen einen Bewertungssplit — es existiert genau *ein*
reales Validierungsbild. TrivialAugment ist zwar suchfrei, setzt aber eine feste
Operationsmenge **auf vorhandenen Bildern** voraus. Diffusionsbasierte
Augmentierung braucht Beispiele der Zielklasse, um sie zu variieren. Und
Albumentations scheitert nicht an „Sichtbarkeit" allgemein, sondern an der
exakten Buchhaltung **über die Malreihenfolge mehrerer eingefügter Objekte**.

**A1–A6 als Scharnier zu Kapitel 5** — mit Substanz, nicht nur den Überschriften:

- **A1 Labelexaktheit** — jede Transformation braucht eine exakt bekannte
  Labeltransformation; boxaufblähende Operationen sind **bildweit verboten** und
  **pro Instanz mit Neuberechnung der Box** auszuführen.
- **A2 Kalibrierte Skala** — erzeugte Größenverteilung gleich gemessener, *und*
  oberhalb der Auflösungsgrenze der feinsten Detektionsebene.
- **A3 Physikalische Zulässigkeit** — nur real mögliche Konfigurationen;
  Invarianzen nur dort, wo sie gelten.
- **A4 Geschlossene Abkürzungen** — für jede Signatur S1–S6 eine Gegenmaßnahme.
- **A5 Identische Labeldefinition** — modale Boxen, dieselben Verwerfungsregeln
  wie die Handannotation (Term λ).
- **A6 Dokumentierte Verteilung** — Parameterbereich *und* tatsächlicher Anteil;
  **Werte ohne Messbeleg sind als Annahme zu kennzeichnen.**

A6 ist eine Selbstbindung, die man laut vorlesen sollte — Block 6 löst sie ein.

---

## Block 5 — Kapitel 4: Datengrundlage (1:00)

**Die Annotationsdefinition zuerst, weil sie alles prägt:**

> Annotiert ist ausschließlich die **sichtbare Stirnfläche** eines
> Rundrohrprofils — heller verzinkter Kranz, dunkle Bohrung. Rohrschäfte in
> Seitenansicht, Gitterknoten und dunkle Löcher im Maschendraht sind **keine**
> Objekte, obwohl sie ähnlich aussehen. Die Box umschließt den **sichtbaren**
> Teil — modal, nicht amodal.

Die erste Festlegung bestimmt, welche harten Negative gebraucht werden; die
zweite, wie der Generator seine Boxen rechnet. Erzeugte der Generator amodale
Boxen, während real modal annotiert ist, entstünde ein systematischer
Größenversatz — ein Verstoß gegen A5. (Merken: In Block 8 stellt sich heraus,
dass genau dort trotzdem ein Rest übrig bleibt.)

**Quellmaterial:** Neun Ausschnitte, 28 × 30 bis 43 × 38 px, sechs mit dunkler
und drei mit heller Bohrung. Sieben Hintergründe, davon **vier Clean Plates** —
Wände aus Rohrschäften *ohne* Stirnflächen, aus derselben Objektart, mit
gemessenen Rohrdurchmessern 16/18/21/20 px. Sie liefern gleichzeitig den
photometrisch passenden Kontext, das Rohmaterial der Okkluder-Bibliothek und
echte Negativbeispiele — und bekommen deshalb sechsfaches Ziehgewicht. Drei
Fremdszenen (Kieswerk, Betonplatten, Betondeck mit Maschendraht) sind primär
hartes Negativmaterial.

**Bewertungsdaten:** `val` 2459 × 1174 mit 88 Objekten für die Modellauswahl,
`test` 1600 × 900 mit 86 Objekten ausschließlich für die Endbewertung.

**Die Messungen, die alles tragen:** Durchmesser Median 27,4 px (val) und
19,6 px (test); NN-Abstand Median 1,09 und 1,24 Objektdurchmesser; 61,4 % bzw.
46,5 % berühren ihren Nachbarn; Größenverhältnis zum Nachbarn Median 1,05.

**Folie:** Tabelle 4.2 auf fünf bis sechs Zeilen gekürzt — Befund → abgeleiteter
Parameter → Belegart. Das ist die Belegbrücke zu Kapitel 5 und macht zugleich
sichtbar, welche Parameter *nicht* durch eine Messung gedeckt sind.

Die Schwäche des Materials selbst nennen: alle vier Clean Plates zeigen dieselbe
Stapelart bei bedecktem bis diffusem Licht — kein harter Sonnenschein, keine
Nässe, kein Schnee.

---

## Block 6 — Kapitel 5: Die Synthese-Pipeline (4:30)

### Leitprinzip (eine Folie, ein Satz)

> Jede Abkürzung schließen, über die ein CNN „hier wurde etwas eingeklebt" statt
> „hier ist ein Rohrende" lernen könnte.

### Arbeitsteilung — folgt zwingend aus A1

Der **Generator** übernimmt alles, was *pro Instanz* wirkt und die Box aus der
transformierten Alphamaske exakt neu bestimmt: Rotation über den vollen Kreis,
Ovalisierung, Delle, Perspektive, Photometrie, Verdeckung, Sichtbarkeitsrechnung.
Die **Trainingszeit** übernimmt alles, was *bildweit* wirkt und Boxen nicht
verfälscht: Mosaic, Skalierung, Translation, Spiegelung, HSV.

Daraus folgt zwingend `degrees = shear = perspective = 0` — **sie blähen
achsparallele Boxen auf**. Die Invarianzen werden also nicht *weggelassen*,
sondern an anderer Stelle und label-exakt erzeugt.

### Zwei Vorstufen — und beide Bausteine werden doppelt genutzt

**Stufe 1, Matting.** Befund 1: Alle neun Ausschnitte sind RGBA-Dateien, aber ihr
Alphakanal ist durchgängig 255 — es sind rechteckige Ausschnitte, bei denen
13–35 % der Fläche nicht zum Objekt gehören. Direktes Einkopieren erzeugte
Signatur S1. Freigestellt wird über ein Ellipsenmodell (c_x, c_y, r_x, r_y, θ):
manuelle Startwerte aus einer 14-fach vergrößerten Rasteransicht, numerische
Verfeinerung per Coordinate Descent auf der radialen Kantenenergie, dann eine
Außenkantenkorrektur bei **25 % statt des klassischen Halbwertspunkts**. Der
Grund ist eine asymmetrische Kostenabwägung: Der helle Kranz ist das
*diskriminative* Merkmal — ihn anzuschneiden entfernt Information, die das Netz
braucht —, während ein etwa 1 px breiter Hintergrundsaum durch Feathering und
Farbangleich ohnehin kaschiert wird. Hart gedeckelt auf [1,02; 1,10]. Ergebnis:
weiche Alphamasken mit rund 1,1 px Übergangsbreite — Absicht, nicht Toleranz,
denn ein 25-px-Objekt hat in einer echten Aufnahme nie eine pixelharte
Silhouette.

**Stufe 2, Okkluder.** Ein synthetisch *gemalter* Balken hätte andere Textur-,
Rausch- und Glanzstatistik als der Rest des Bildes — das Netz könnte „Objekt
liegt neben künstlichem Balken" lernen, also selbst eine Abkürzung der Klasse
S2–S4. Die vier Clean Plates sind dagegen Wände aus Rohrschäften **desselben
Materials, derselben Kamera und derselben Beleuchtung**. Daraus gewonnen:
dominante Rohrrichtung über den Struktur-Tensor (gemessen +0,9, −1,6, +0,9,
+6,6 Grad), Hochpassfilterung des Zeilenmittelprofils zur Bestimmung der
Rohrachsen — ihr Abstand *ist* der Rohrdurchmesser, 16/18/21/20 px, dieselben
Werte, die den Maßstab verankern. Ergebnis: **56 echte Rohrschäfte** mit echter
zylindrischer Schattierung und echtem Glanzlicht.

**Die Doppelnutzung ist ein eigener Punkt wert:** Die sieben Hintergründe sind
Quelle der Okkluder-Bibliothek *und* Bildhintergrund *und* Fremdkörperquelle. Die
56 Schäfte dienen als Schaft **hinter** jeder Stirnfläche (T5) *und* als Okkluder
**vor** dem Stapel (T6). Aus 16 Eingabedateien entsteht alles.

### Stufe 3 — T1 bis T9

Leinwand 1024 × 1024, identisch zur Trainingsauflösung: Die synthetischen Bilder
durchlaufen im Training **keine** Größenänderung, die kalibrierte
Objektgrößenverteilung bleibt exakt erhalten (A2).

**Folie:** Tabelle 5.1, ein Satz je Stufe. Dazu `abb:stufen` — die sechs
Teilbilder, die dasselbe Bild Stufe für Stufe aufbauen. Das ist die überzeugendste
Abbildung der Arbeit.

Von den fünf Modellierungsentscheidungen reichen zwei für den Vortrag:

**Maßstabskopplung.** Der Zoomfaktor wird so gezogen, dass der Basisdurchmesser
im Zielbereich [12, 44] px liegt: d_Basis = d_Rohr · z. Objektgröße und
Hintergrundmaßstab sind damit **nicht unabhängig** gezogen — die Stirnfläche hat
den Durchmesser des Schafts, an dem sie sitzt. Es ist dasselbe Rohr. Das ist der
Kern von A3.

**Sichtbarkeitsrechnung (T7) — das methodische Herzstück.** Kein
Copy-Paste-Generator darf Labels blind vergeben: Ein Objekt, das nach dem
Einfügen zu 95 % verdeckt ist, hat kein sinnvolles Label mehr. Für *jede* Instanz
wird deshalb die exakte Sichtbarkeit nach Porter/Duff gerechnet,

    α_i^sichtbar = α_i · ∏_{j>i} (1 − α_j)

über **alle** später gemalten Schichten — andere Profile, Rohrschäfte,
Fremdkörper. Die Buchhaltung läuft über die beschnittenen Alphafenster in
Malreihenfolge, ist also exakt und nicht approximiert. Daraus vier Regeln, und
zwar **dieselben, nach denen val und test annotiert wurden** (A5): verworfen bei
unter 30 % Sichtbarkeit (30 801 Objekte), bei unter 15 % der Fläche im Bild
(3 846) und bei sichtbarer Box unter 5 px (228); vergeben wird stets die
**modale** Box. Insgesamt **34 875 von 173 100 Instanzen verworfen — 20,1 %**.

> Diese Zahl ist kein Verlust, sondern der Ertrag der Stufe: Sie markiert genau
> jene Bildstellen, an denen ein naiver Generator ein falsches Label gesetzt
> hätte — und *kein* Label macht dieselbe Bildstelle zum korrekten harten
> Negativ.

Die übrigen drei Entscheidungen in je einem Nebensatz, falls Zeit bleibt: Die
Ovalisierung erhält den geometrischen Mittelwert, weil die Realmessung
√Fläche verwendet. Der Farbangleich ist bewusst *partiell* — ein vollständiger
Reinhard-Transfer ebnete den Albedo-Unterschied zwischen Bohrung und Kranz ein
und zerstörte damit das Objekt; modelliert wird nur die Beleuchtungs- und
Weißabgleichdifferenz. Und das Layoutmodell braucht seinen Rauschterm, weil sonst
sichtbar künstliche Reihen entstehen.

### Der erzeugte Datensatz

6000 Bilder à 1024², davon 583 Negativbilder (9,7 %), mit **136 384 gelabelten
Objekten** (22,7 pro Bild) aus 173 100 gerenderten Instanzen. Mittlere
Sichtbarkeit 90,0 %, 30,1 % teilverdeckt. Durchmesser p05/Median/p95 =
11,6/22,5/39,9 px.

### Der Qualitätsnachweis ist nicht der optische Eindruck

**Folie:** Tabelle 5.2 — die erzeugte Statistik gegen die gemessene.

| Kenngröße | real val | real test | synthetisch |
|---|---|---|---|
| NN-Abstand p10/Median/p90 | 0,86 / 1,09 / 1,78 | 0,83 / 1,24 / 2,51 | **0,84 / 1,17 / 1,87** |
| berührt den Nachbarn | 61,4 % | 46,5 % | **53,6 %** |
| Nachbarn innerhalb 5 Ø | 10 | 10 | **13** |
| Größenverhältnis zum Nachbarn | 1,05 | 1,05 | **1,05** |

Die synthetische Verteilung liegt bei allen vier Kenngrößen zwischen oder nahe
bei den beiden realen Messungen. Die einzige systematische Abweichung ist die
Nachbarzahl — bewusst als Robustheitsreserve belassen, weil sie den Detektor auf
schwierigere Szenen vorbereitet als die, die er zur Inferenzzeit vorfindet. Die
umgekehrte Abweichung wäre gefährlich gewesen.

**Ein Satz, der eine sichere Rückfrage vorwegnimmt:** Alle genannten Anteile sind
**keine konfigurierten Sollwerte, sondern gemessene Ist-Werte** aus
`synth/train/generation_report.json` über alle 6000 Bilder. Erst das macht A6
überprüfbar.

### Der Bogen schließt sich

A4 verlangte für jede Signatur eine Gegenmaßnahme — und hier steht sie:
S1 → Alphamaske und Kantenweichzeichnung · S2 → partieller Farbangleich ·
S3 → flächentreue Skalierung und Unschärfeangleich · S4 → lokal geschätzter
Rauschangleich · S5 → JPEG-Requantisierung · S6 → Schlagschatten und
physikalische Verdeckung.

Und die Bilanz, die A6 einlöst: **36 Transformationen mit 40 Parameterwerten —
15 gemessen, 10 literaturgestützt, 6 physikalisch begründet, 2 gemischt,
7 ausdrücklich als Annahme gekennzeichnet.** 82 % auf eine der drei zulässigen
Quellen zurückgeführt, 18 % Annahmen, einzeln benannt.

---

## Block 7 — Kapitel 6: Trainingsaufbau (0:30)

Halbe Folie, drei Sätze:

`yolo11s`, 9,4 Millionen Parameter, von COCO-Gewichten — klein, weil der
Datensatz eine Klasse und nur neun zugrunde liegende Objektinstanzen hat.
`imgsz` = 1024, identisch zur Leinwand: Die Objekte erreichen das Netz mit exakt
der Pixelgröße, für die die Pipeline ausgelegt ist.

**Die wichtigste Entscheidung des Kapitels:** Validiert wird auf dem **realen**
val-Bild, nicht synthetisch. Der Trainingsverlust fällt über alle 30 Epochen
monoton, die reale Metrik ist ab Epoche 10 flach — ab dort verbessert das Modell
ausschließlich seine Anpassung an Eigenschaften der synthetischen Daten, die in
der Zieldomäne nicht gelten. Eine synthetische Validierung hätte das Modell aus
der letzten Epoche gewählt und damit das am stärksten überangepasste.

---

## Block 8 — Kapitel 7: Ergebnisse (4:00)

### Zuerst das Messinstrument (20 s — es lohnt sich)

Leitgröße ist **mAP@0,5**, ergänzt um das beste F₁ über alle Konfidenzschwellen
samt der zugehörigen Schwelle c\* — dem Betriebspunkt, den man in einer
Anwendung wählen würde.

Die Bewertung läuft über eine eigene Implementierung, weil die mittlere
Treffer-IoU und der F₁-optimale Betriebspunkt gebraucht werden, die die
Standardfunktion nicht ausgibt — gegen die Referenzimplementierung geprüft, max.
1,3 Prozentpunkte Abweichung.

**Die Lesevorschrift für das ganze Kapitel — dieser Satz muss fallen:**
mAP@0,5:0,95 ist bei dieser Objektgröße weniger ein Maß für die
Detektionsqualität als für Pixelauflösung und Labelkonvention. Das ist der Grund,
warum mAP@0,5 die Leitgröße ist, und es gilt für jede folgende Tabelle.

### Hauptergebnis

| Split | mAP@0,5 | mAP@0,75 | mAP@0,5:0,95 | F₁ | c\* | P / R | mittl. IoU |
|---|---|---|---|---|---|---|---|
| val (real) | **0,886** | 0,793 | 0,597 | 0,862 | 0,336 | 0,872 / 0,852 | 0,824 |
| test (real) | **0,956** | 0,326 | 0,466 | 0,971 | 0,263 | **1,000** / 0,944 | 0,746 |
| sval (synth.) | 0,982 | 0,935 | 0,828 | 0,964 | 0,450 | 0,972 / 0,956 | 0,899 |

Ein ausschließlich auf synthetischen Bildern trainiertes Modell erreicht auf
echten, nie im Training gesehenen Aufnahmen **mAP@0,5 von 0,886 und 0,956**; am
F₁-optimalen Betriebspunkt findet es 85 % (val) bzw. 94 % (test) der Objekte —
und auf test sitzt **jede** Detektion auf einem annotierten Objekt
(Precision 1,000). Das ist die Antwort auf die Aufgabenstellung. Mit
Test-Time-Augmentation steigt mAP@0,5 auf test auf 0,963.

### Sim-to-Real-Abstand — die Argumentation ist wichtiger als die Zahl

0,982 gegen 0,886 und 0,956, also **9,6 bzw. 2,6 Prozentpunkte** mAP@0,5.

Dazu steht die Ben-David-Schranke (Gl. 3.6/3.7) mit Werten besetzt im Text:
ε_S = 0,018, ε_T = 0,044, daraus `½·d + λ ≥ 0,026`. Der Größenversatz trägt dazu
nichts bei — ein Angleichen der Boxgröße verbessert mAP@0,5 nicht, sondern senkt
es leicht; bei Zuordnungsschwelle 0,5 ist die Konventionsdifferenz wirkungslos.
Die Gegenprobe (Netz lernt Artefakte) ergäbe `½·d + λ ≥ 0,94`, den **36-fachen**
Wert. **Das ist die stärkste
Folie des Blocks** — sie macht aus „klein genug / groß genug" eine Rechnung.

> **Klein genug**, dass das Verfahren praktisch funktioniert, und **groß genug**,
> um zu belegen, dass das Netz *nicht* bloß Einfügeartefakte gelernt hat: Wäre
> die Pipeline an einer der Signaturen S1–S6 durchlässig, wäre sval nahezu
> perfekt (> 0,99) und die realen Bilder katastrophal. Genau dieses Muster tritt
> nicht auf.

Der einzige *direkte* empirische Hinweis, dass die Anti-Abkürzungs-Maßnahmen
greifen — und ein Indiz, kein Beweis. Diese Einschränkung selbst nennen.

### Die Inferenzauflösung (kein Experiment — kein Modell wird neu trainiert)

Von `imgsz` 640 auf 1600 steigt mAP@0,5 auf val von 0,435 auf 0,904: **47
Prozentpunkte, der größte gemessene Einzeleffekt der Arbeit**, größer als jede
einzelne Entscheidung in der Pipeline.

Und er war **vorhergesagt**: Bei 640 wird das 2459 px breite val-Bild um den
Faktor 0,26 verkleinert, die Objekte schrumpfen von 20–36 auf 5–9 px und
unterschreiten die Schrittweite 8. Der Rückruf bricht auf 0,398 ein — das Modell
*findet* die Objekte nicht mehr, es lokalisiert sie nicht bloß schlechter.

Bemerkenswert: Das Optimum liegt **nicht** am nativen Maßstab — auf val liegt es
bei 1600 und damit klar unter den nativen 2464 (0,904 gegen 0,886). Auf test sind
nativer Maßstab und 1920 gleichwertig (0,956 gegen 0,956); die frühere
Verbesserung durch Hochskalieren war ein Artefakt der alten Annotation. Der
Grund für die Richtung: Der Generator erzeugt Objekte mit Median 22,5 px, test
hat nur 17,0 px. → **Nicht auf den nativen Maßstab skalieren, sondern so, dass die
erwartete Objektgröße im Netzeingang dem Median der Trainingsverteilung
entspricht.** Objektgrößenbereich des Generators und Inferenzauflösung sind nicht
unabhängig voneinander wählbar.

Zur Sauberkeit sagen: Ausgewählt wurde nach höchstem mAP@0,5:0,95 **auf val
allein**; das Kriterium stand **vor den Läufen fest** und gilt unverändert für
alle ausgewerteten Modelle. Eine Wahl je Split wäre Hyperparameterselektion auf
dem Testsatz.

### Das Experiment: Wie viel Verdeckung in den Trainingsdaten? (die zentrale Folie)

**Frage.** Die Okklusionserzeugung ist die aufwendigste Einzelstufe der Pipeline
— sie erfordert die Okkluder-Bibliothek, das Tiefenmodell und die exakte
Sichtbarkeitsrechnung. Zahlt sich der Aufwand messbar aus?

**Aufbau.** A0 hat keine quer liegenden Stangen, keine Fremdkörper und keinen
Schaft, der vor einer Stirnfläche kreuzt. A1 ist die Baseline. A2 hat in jedem
Bild Stangen, mehr Fremdkörper und mehr quer vor dem Stapel liegende Rohre. **Der
Rohrschaft *hinter* der Stirnfläche bleibt in allen Armen erhalten** — er ist
physikalische Randbedingung, keine Augmentierung. (Diesen Satz sagen, er nimmt
die naheliegendste Rückfrage vorweg.)

**Die Kontrolle ist besser als bei einer gewöhnlichen Datensatz-Ablation:** Weil
der Seed identisch ist, rendern alle drei Arme **genau dieselben 86 617 Instanzen
an denselben Positionen**. Der einzige Unterschied ist, wie viel davon verdeckt
wird — und als Folge, wie viele Labels die Sichtbarkeitsregel verwirft.

| Arm | Labels | Sichtbarkeit | teilverd. | test mAP@0,5 | test mAP@0,5:0,95 | test F₁ |
|---|---|---|---|---|---|---|
| A0 ohne | 82 950 | 0,977 | 12,0 % | 0,9577 | 0,441 | 0,928 |
| A1 Baseline | 68 281 | 0,902 | 29,9 % | 0,9584 | **0,482** | 0,949 |
| A2 stark | 55 833 | 0,852 | 40,7 % | **0,9596** | 0,479 | **0,959** |

**Drei Befunde:**

1. **Der Effekt ist überwiegend monoton** — auf val +4,2 PP mAP@0,5 und
   +1,4 PP mAP@0,5:0,95, auf test +3,1 PP F₁, jeweils über alle drei Stufen in
   derselben Richtung. **Zwei Einschränkungen offen ansprechen:** test mAP@0,5
   ist mit 0,19 PP Spanne gesättigt, und test mAP@0,5:0,95 ist nicht monoton
   (A1 über A2). Vier von fünf Spalten, nicht fünf.
2. **Der Gewinn entsteht gegen einen Verlust an Supervision.** A2 hat **32,7 %
   weniger Labels** als A0, bei identischer Bildzahl und identischen gerenderten
   Instanzen. Wären die zusätzlichen Labels wertvoller als die zusätzliche
   Schwierigkeit, müsste A0 gewinnen. Das ist der stärkste Einzelbeleg dafür,
   dass die Okklusionsstufe kein Beiwerk ist.
3. **Ohne Verdeckung ist die Aufgabe zu leicht.** A0 erreichte sein bestes Modell
   bereits in Epoche 3 und verbesserte sich in den folgenden 20 Epochen nicht
   mehr; A1 und A2 verbesserten sich bis zur letzten Epoche 30 durchgehend.

**Die Einschränkungen gleich mitliefern** — das nimmt die Frage vorweg und wirkt
souveräner, als sie zu kassieren: Der Effekt liegt *innerhalb* des
Konfidenzintervalls von ±0,05 (val) bzw. ±0,04 (test). Belastbar ist nicht die
Einzeldifferenz, sondern die **Konsistenz** — dieselbe Richtung in allen fünf
Metrikspalten über drei Stufen, was bei reinem Rauschen unwahrscheinlich wäre. mAP@0,75 auf
val verläuft nicht monoton (0,812 / 0,760 / 0,775); die Aussage betrifft das
*Finden* der Objekte, nicht die Präzision der Boxen. Labelzahl und
Okklusionsanteil sind nicht entkoppelt. Und A1 und A2 waren nach 30 Epochen nicht
auskonvergiert — die Aussage ist konservativ, aber nicht abgeschlossen.

### Der Größenversatz der Boxen — der beste Teil der Arbeit (60 s)

**Die Beobachtung:** mAP@0,75 liegt auf test bei **0,326** — neben einem
mAP@0,5 von 0,956. Das Modell *findet* praktisch jedes Objekt, trifft seine
Boxgröße aber systematisch zu klein.

**Und die Gegenprobe, die daraus ein Diagnosewerkzeug macht statt einer
Metrikschönung: Auf val ändert dieselbe Absenkung nichts.** Berichtet wird weiter
bei 0,50, dem Bezugswert von PASCAL VOC und COCO.

**Die Messung:**

| Split | Paare | Objekt-Ø (Median) | pred/GT | GT − pred |
|---|---|---|---|---|
| test (real) | 82 | 22,3 px | 0,884 | +2,53 px |
| val (real) | 75 | 27,5 px | 0,961 | +1,16 px |
| sval (synthetisch) | 1064 | 22,0 px | **1,012** | −0,29 px |

**Der Schluss:** Auf dem synthetischen Split trifft das Modell die Boxgröße auf
1 % genau. Die Ursache liegt also **nicht in der Boxregression, sondern in der
Labelkonvention** — die Generatorboxen folgen der α > 0,5-Kontur der Alphamaske,
die Handannotation umschließt großzügiger. A5 ist damit nur *annähernd* erfüllt;
der Restversatz beträgt messbar 1,2 bis 2,3 px. Der Zentrumsversatz liegt bei
rund 0,05 Objektdurchmessern — es ist ein Größen-, kein Positionsfehler.

**Was das kostet:** Weitet man alle vorhergesagten Boxen um den gemessenen
medianen Fehlbetrag auf (1,15 px auf test) und wertet neu aus, steigt **mAP@0,75
von 0,326 auf 0,752** und mAP@0,5:0,95 von 0,466 auf 0,617 — ohne dass am Modell
etwas geändert wird; mAP@0,5 und F₁ sinken dabei sogar leicht. Die berichteten Lokalisierungswerte sind also
**pessimistisch**: Sie messen zu einem erheblichen Teil eine Konventionsdifferenz
und nicht die Lokalisierungsgüte. Auf mAP@0,5 und F₁ — die hier maßgeblichen
Größen — bleibt der Effekt klein.

> **Die Verallgemeinerung:** Bei programmatisch erzeugten Labels wird die
> Labelkonvention selbst zu einer messbaren Größe.

### Statistische Belastbarkeit (10 s, aber nennen)

Je Split liegt nur *ein* Bild vor, ein gewöhnlicher Bootstrap über Bilder ist
also nicht möglich. Verwendet wird ein räumlicher Block-Bootstrap über Kacheln
mit 400 Wiederholungen — Blockbildung, weil benachbarte Objekte Beleuchtung,
Schärfe und Verdeckungssituation teilen: val 0,892 ± 0,051, test 0,943 ± 0,023.

---

## Block 9 — Kapitel 8: Bilanz und Grenzen (1:00)

**Die ehrliche Bilanz, ungefragt.** Von 40 Parameterwerten ist **einer**
kontrolliert abliert. Von 36 Transformationen sind drei durch die Ablation
gemessen, sechs an der Realverteilung kalibriert, 27 begründet, aber nicht
einzeln belegt. Die Begründungssystematik trägt den Entwurf; der experimentelle
Nachweis trägt einen Punkt daraus. — Das schreibt niemand freiwillig hin, und
genau deshalb ist es der glaubwürdigste Teil der Arbeit. Sagen Sie es zuerst.

**Die unbequeme Bilanz.** Der größte gemessene Einzeleffekt ist *keine*
Augmentierungsentscheidung, sondern die Inferenzauflösung. Das relativiert den
Aufwand der Pipeline nicht — ohne sie gäbe es überhaupt keinen Trainingsdatensatz
—, verschiebt aber die Priorität: **Wer eine solche Anwendung baut, sollte die
Auflösungsfrage zuerst klären und erst danach die Feinheiten der Synthese
optimieren.**

**Das positive Gegengewicht — eine Größe, vier Entscheidungen.** d_eff = d · M/N
erklärt (1) die Generatorkalibrierung auf 12–44 px, (2) die Trainingsauflösung
1024, (3) die Inferenzauflösung 1600–1920 und (4) den Einbruch auf 0,435 bei
`imgsz` 640 — der vorhergesagt war, bevor er gemessen wurde. Eine gemessene
Größe, vier abgeleitete Entscheidungen, eine eingetroffene Vorhersage. Das ist
das methodisch belastbarste Ergebnis.

**Was das Experiment nicht beantwortet** — selbst sagen, bevor gefragt wird: ob
die *physikalische Plausibilität* der Verdeckung nötig ist. Ein Arm mit gleich
starker, aber generischer Verdeckung (etwa Cutout) hätte das entschieden und
fehlt. Damit bleibt die aufwendigste Begründung der Pipeline — echte Rohrschäfte
statt gemalter Balken — unbelegt.

**Die schwerwiegendste Grenze:** je ein Bild pro Split. Daraus Konfidenz­
intervalle von ±0,04 bis ±0,05; beide Bilder zeigen dieselbe Objektart bei
ähnlichem Wetter; und val wird doppelt verwendet — für die
Modellauswahl *und* für die Wahl der Inferenzauflösung. **Belastbar sind die
test-Zahlen.**

**Die drei wichtigsten nächsten Schritte:** (0) Labelkonvention angleichen — eine
geänderte Boxschwelle im Generator genügt, rund 39 Punkte mAP@0,75 sind zu
erwarten, der billigste große Gewinn. (1) Zehn bis zwanzig annotierte Realbilder
— sie brächten die Konfidenzintervalle auf ein Maß, bei dem das Experiment
entscheidbar wäre. (2) Die Auflösungsfrage architektonisch angehen: höhere
Inferenzauflösung gegen einen feineren Detektionskopf auf Schrittweite 4 gegen
kachelweise Inferenz, bei gleichem Speicherbudget.

**Schlussfolie:** die fünf Einsichten, nur die Titel plus je eine Zahl. (Punkt 4
vorher anpassen — siehe unten.)

---

## Rückfragen, die kommen werden

**„Sie wählen die Inferenzauflösung nach mAP@0,5:0,95 auf val — zeigen als
Hauptergebnis aber den nativen Maßstab, der schlechter ist."**
Die unangenehmste Frage, und sie steht so in Ihrer eigenen Tabelle 7.2. Die
Antwort hat zwei Teile. Erstens ist der native Maßstab der einzige
*parameterfreie* Bezugspunkt — er zeigt, was das Modell ohne jede Nachjustierung
leistet, und der Wert bei gewählter Auflösung (0,928 auf test) steht direkt
daneben im Text; versteckt ist nichts. Zweitens ist das Auswahlkriterium selbst
angreifbar, und das sagen Sie besser vor dem Prüfer: mAP@0,5:0,95 misst bei
diesen Objektgrößen zu einem erheblichen Teil die Labelkonvention
(Abschnitt 7.7). Nach mAP@0,5 ausgewählt fiele die Wahl auf 1600 statt 1920 —
beide Werte stehen in der Tabelle. Siehe Punkt 2 in „Vor dem Vortrag"; daraus
lässt sich eine eigene Folie machen.

**„Warum nur ein Experiment?"**
Weil es das einzige ist, dessen unabhängige Variable die eigene Pipeline ist. Der
Auflösungssweep braucht kein Neutraining und ist deshalb kein Experiment, sondern
Teil des Bewertungsprotokolls. Ein Architekturvergleich hätte über Augmentierung
nichts ausgesagt.

**„Ist A0 dann überhaupt noch Ihr Objekt?"**
Der Schaft hinter der Stirnfläche bleibt in allen drei Armen — er ist
physikalische Randbedingung, keine Augmentierung. Variiert wird ausschließlich,
was *vor* dem Objekt liegt.

**„Neun Objekte, jedes rund 15 100-mal — lernt das Netz die nicht auswendig?"**
Das ist die ehrliche Grenze, und sie steht in Kapitel 8. Das Gegenargument ist
der Sim-to-Real-Abstand: Hätte das Netz die neun Ausschnitte memoriert, wäre sval
nahezu perfekt und real katastrophal. Zusätzlich ist das Modell bewusst klein
gewählt.

**„Warum keine generative Augmentierung, Diffusionsmodelle?"**
Steht seit dem letzten Stand in Kapitel 3: Sie braucht Beispiele der Zielklasse,
um sie zu variieren — genau das fehlt.

**„Ihr Effekt liegt im Konfidenzintervall."**
Ja, und das steht so in der Arbeit. Tragend ist die Konsistenz über 3 Metriken ×
2 Splits × 3 Arme, nicht die Einzeldifferenz.

**„Warum COCO-Vortraining bei einer so fremden Domäne?"**
Die Low-Level-Filter sind domänenunabhängig, und Hinterstoisser zeigt den Nutzen
explizit für synthetische Trainingsdaten.

**„Warum nicht Albumentations?"**
Nicht wegen Sichtbarkeit allgemein — wegen der exakten Buchhaltung über die
Malreihenfolge mehrerer eingefügter Objekte. Die liegt außerhalb des
Funktionsumfangs.

---

## Vor dem Vortrag zu klären

1. **Fazit-Punkt 4 zitiert ein Ergebnis, das nicht mehr im Dokument steht.**
   `08_diskussion.tex:174–179` nennt „Wahl des Detektionskopfes" und den
   Vorsprung „4,6 gegenüber 0,4 Prozentpunkte" — diese Messung liegt im
   `\iffalse`-Block von Kapitel 7. Entweder umformulieren (die Rechnung sagt
   *vorher*, wo ein feinerer Kopf wirken würde — als Ausblick) oder Punkt 4 auf
   die drei belegten Ebenen kürzen. So wie es steht, verweist das Fazit auf eine
   Zahl, die der Leser nicht findet.

2. **Die Einsicht „das Auswahlkriterium entscheidet mit" lässt sich retten.**
   Sie hing bisher am Zählfehler und ist mit ihm entfallen — dasselbe Phänomen
   steckt aber in Tabelle 7.2, die im Dokument geblieben ist: Auf val ist
   **mAP@0,5 bei `imgsz` 1600 maximal (0,904), mAP@0,5:0,95 dagegen bei 1920**.
   Zwei Detektionsmetriken, zwei verschiedene Optima, und die Arbeit wählt
   begründet nach mAP@0,5:0,95. Ein Absatz in 7.3 und eine Folie — die Aussage
   „welche Metrik man ansetzt, ist eine Entscheidung und keine Formalität" steht
   damit ohne jeden Zählbezug auf eigenen Füßen.
   (Geprüft gegen `docs/experiments/_yolo11s_backup/experiment1_inference.json`.)

3. **`05_pipeline.tex:198` gegen `:200–201`:** Teilbild (d) nennt 82 Instanzen,
   Teilbild (f) 48 vergebene plus 33 verworfene — das sind 81. Auf einer Folie
   liefert das eine Nachfrage frei Haus. Entweder klären oder in der
   Bildunterschrift nur „grün bekommt ein Label, rot nicht" sagen.

4. ~~**`07_auswertung.tex:69`: „Gefunden werden dabei etwa 88–89 % der
   Objekte."**~~ **Erledigt** (31.08): Der Satz lautet jetzt „findet es 85 bis
   88 % der Objekte bei einer Precision von 0,87 bis 0,91" und stimmt damit gegen
   die Recall-Spalte der Tabelle darüber.

5. ~~**Sim-to-Real:** Rechenfehler in der Prozentpunktangabe.~~ **Erledigt**
   (31.08): Der Abstand beträgt 9,6 (val) und 2,6 (test) Prozentpunkte; beide
   Zahlen stehen so im Text.

6. **Reproduzierbarkeit:** Die Zahlen der Arbeit stammen vom `yolo11s`-Lauf und
   liegen in `docs/experiments/_yolo11s_backup/`. Unter dem Pfad, den README und
   Anhang nennen (`docs/experiments/experiment1_inference.json`), stehen
   inzwischen andere Werte aus einem `yolo26s`-Lauf — bei `imgsz` 640
   beispielsweise 0,420 statt 0,435. Wer nachrechnet, bekommt Abweichungen. Ich
   habe Tabelle 7.2 vollständig gegen das Backup geprüft: sie stimmt. Es fehlt
   nur der Hinweis, welche Datei gemeint ist.
