# 2 Theoretische Grundlagen

## 2.1 Convolutional Neural Networks (CNNs)

### 2.1.1 Motivation und Grundidee

Convolutional Neural Networks (CNNs, dt. Faltungsnetzwerke) sind eine spezielle Klasse künstlicher neuronaler Netze, die für die Verarbeitung von Daten mit gitterartiger Topologie – insbesondere Bildern – entwickelt wurden (Goodfellow et al., 2016). Während klassische vollvernetzte Netze (Multilayer Perceptrons) jedes Pixel eines Bildes mit jedem Neuron der Folgeschicht verbinden und dadurch bei hochauflösenden Bildern eine unpraktikabel große Zahl an Parametern erzeugen, nutzen CNNs drei zentrale Prinzipien, um diese Problematik zu umgehen: lokale rezeptive Felder, geteilte Gewichte (weight sharing) und räumliches Subsampling (LeCun et al., 1998).

Die Grundannahme dahinter ist, dass relevante Bildmerkmale – etwa Kanten, Ecken oder Texturen – lokal auftreten und translationsinvariant sind: Eine Kante ist eine Kante, unabhängig davon, an welcher Bildposition sie erscheint. Ein gelerntes Filter kann daher über das gesamte Bild hinweg wiederverwendet werden.

### 2.1.2 Die drei Grundprinzipien im Detail

Die drei genannten Prinzipien sind keine Implementierungsdetails, sondern kodieren strukturelles Vorwissen über Bilddaten in die Netzarchitektur (sogenannter *induktiver Bias*). Sie werden im Folgenden formal hergeleitet.

**Notation.** Ein Grauwertbild wird als Funktion $x: \Omega \to \mathbb{R}$ auf dem diskreten Gitter $\Omega = \{0,\dots,H-1\} \times \{0,\dots,W-1\}$ aufgefasst; $x[i,j]$ bezeichnet den Wert an Position $(i,j)$. Eine Netzschicht ist eine Abbildung, die eine Eingabe mit $C_{\text{in}}$ Kanälen auf eine Ausgabe mit $C_{\text{out}}$ Kanälen überführt.

**Lokale rezeptive Felder.** In einer vollvernetzten Schicht berechnet sich die Aktivierung eines Neurons $k$ aus *allen* Eingangspixeln:

$$y_k = \sigma\!\left(\sum_{(i,j)\in\Omega} w_k[i,j]\,x[i,j] + b_k\right)$$

Die Parameterzahl einer solchen Schicht beträgt $C_{\text{out}} \cdot H \cdot W + C_{\text{out}}$. Für $H = W = 640$ und lediglich $C_{\text{out}} = 100$ Neuronen ergeben sich bereits über 40 Millionen Gewichte. Neben dem unmittelbaren Speicher- und Rechenaufwand ist dabei vor allem die **Stichprobeneffizienz** problematisch: Ein vollvernetztes Netz besitzt keinerlei architektonisches Vorwissen darüber, dass dasselbe visuelle Muster an unterschiedlichen Bildpositionen dieselbe Bedeutung trägt, und müsste diese Eigenschaft für jede Position separat aus Daten erlernen. Der entsprechende Bedarf an Trainingsbeispielen ist in der Praxis kaum zu decken.

![](./rezeptives_feld.jpeg)

Die klassische statistische Lerntheorie führt hohe Parameterzahlen darüber hinaus auf eine erhöhte Modellkapazität und damit auf eine Neigung zur Überanpassung zurück (Goodfellow et al., 2016, Kap. 5). Für tiefe neuronale Netze ist dieser Zusammenhang allerdings nicht ungebrochen gültig: Zhang et al. (2017) zeigen experimentell, dass gängige Faltungsnetze zufällige Beschriftungen der Trainingsdaten vollständig memorieren können und dennoch auf realen Daten gut generalisieren – ein Befund, der die klassische Kapazitätsargumentation als alleinige Erklärung in Frage stellt. Der belastbare Vorteil lokaler rezeptiver Felder liegt daher weniger in einer Vermeidung von Überanpassung an sich als in der beschriebenen Stichprobeneffizienz und im deutlich reduzierten Rechenaufwand.

In einer Faltungsschicht wird die Summation stattdessen auf eine lokale Nachbarschaft $\mathcal{N}_{k \times k}$ der Größe $k \times k$ beschränkt:

$$y[i,j] = \sigma\!\left(\sum_{(u,v)\in\mathcal{N}_{k\times k}} w[u,v]\,x[i+u,\,j+v] + b\right)$$

Die Anzahl der Verbindungen je Ausgabeneuron sinkt damit von $H \cdot W$ auf $k^2$ – für $k = 3$ von 409.600 auf 9. Die Rechtfertigung liegt in der lokalen Korrelationsstruktur natürlicher Bilder: Die Autokorrelation benachbarter Pixelwerte fällt mit wachsendem Abstand rasch ab, sodass weit entfernte Pixel für lokale Merkmalsentscheidungen kaum Information beitragen (LeCun et al., 1998).

Das theoretische rezeptive Feld wächst mit der Netztiefe. Bezeichnet $r_\ell$ seine Kantenlänge nach Schicht $\ell$, $k_\ell$ die Kernelgröße und $s_\ell$ die Schrittweite, so gilt die Rekursion

$$r_\ell = r_{\ell-1} + (k_\ell - 1)\prod_{m=1}^{\ell-1} s_m, \qquad r_0 = 1.$$

Bei reinen $3\times3$-Faltungen mit $s_m = 1$ wächst $r_\ell$ lediglich linear ($r_\ell = 2\ell + 1$); erst durch Schrittweiten $s_m = 2$ wächst das Produkt exponentiell, sodass das rezeptive Feld schnell große Bildbereiche umfasst. Hieraus entsteht die in Abschnitt 2.1.4 beschriebene Merkmalshierarchie. Zu beachten ist, dass das *effektive* rezeptive Feld kleiner ausfällt als dieses theoretische Maß: Luo et al. (2016) zeigen, dass der Einfluss $\partial y_{\text{Zentrum}} / \partial x[i,j]$ annähernd gaußförmig zum Zentrum hin abfällt und nur ein Bruchteil des theoretischen Felds tatsächlich zur Ausgabe beiträgt.

**Geteilte Gewichte (Weight Sharing).** Ohne Gewichtsteilung besäße jede Position $(i,j)$ einen eigenen Kernel $w_{ij}$, was auf $H \cdot W \cdot k^2$ Parameter je Merkmalskarte führte. Weight Sharing bedeutet formal, dass $w_{ij} = w$ für alle $(i,j)$ gilt; die Schicht wird dadurch zur diskreten Faltung (genauer: Kreuzkorrelation)

$$y[i,j] = (w * x)[i,j] = \sum_{u,v} w[u,v]\,x[i+u,\,j+v],$$

mit lediglich $k^2$ Parametern je Kernel. Für eine Schicht mit $C_{\text{in}}$ Eingangs- und $C_{\text{out}}$ Ausgangskanälen ergeben sich $C_{\text{out}} \cdot C_{\text{in}} \cdot k^2 + C_{\text{out}}$ Parameter – unabhängig von der Bildgröße. Dies ist der entscheidende Unterschied zur vollvernetzten Schicht, deren Parameterzahl mit $H \cdot W$ skaliert.

Die zentrale Eigenschaft ist die **Translationsäquivarianz**. Bezeichnet $(T_\tau x)[i,j] = x[i-\tau_1,\,j-\tau_2]$ die Verschiebung um $\tau$, so gilt für die Faltungsoperation $f$:

$$f(T_\tau x) = T_\tau f(x).$$

Eine Verschiebung der Eingabe führt also zu einer identischen Verschiebung der Ausgabe-Merkmalskarte. Die Rechtfertigung ist die (näherungsweise) Stationarität natürlicher Bildstatistik: Die relevanten lokalen Merkmale sind ortsunabhängig – eine Kante bleibt eine Kante, unabhängig von ihrer Position. Zu betonen ist, dass Äquivarianz nicht Invarianz bedeutet; Letztere entsteht erst durch das nachfolgend beschriebene Subsampling.

Aus der Gewichtsteilung folgen drei praktische Konsequenzen: erstens eine erhöhte Dateneffizienz, da ein Muster, das nur an wenigen Positionen in den Trainingsdaten vorkommt, dennoch überall erkannt wird; zweitens eine Einschränkung des Hypothesenraums, da die Nebenbedingung $w_{ij} = w$ nur noch translationsäquivariante Abbildungen zulässt – eine Beschränkung, die dem tatsächlichen Charakter der Aufgabe entspricht und in diesem Sinne als struktureller Prior wirkt; drittens die Notwendigkeit mehrerer Kernel je Schicht, da ein Kernel genau ein Muster repräsentiert (LeCun et al., 1998; Goodfellow et al., 2016).

**Räumliches Subsampling.** Für die Objekterkennung ist häufig *Invarianz* gegenüber kleinen Verschiebungen erwünscht, also $f(T_\tau x) \approx f(x)$ für kleine $\|\tau\|$. Diese entsteht durch stufenweise Reduktion der räumlichen Auflösung – klassisch durch Pooling über ein Fenster $\mathcal{P}$ der Größe $p \times p$ mit Schrittweite $s$:

$$y[i,j] = \max_{(u,v)\in\mathcal{P}} x[s\cdot i + u,\; s\cdot j + v] \quad \text{(Max-Pooling)}$$

In modernen Architekturen wie YOLO26 wird dieselbe Auflösungsreduktion überwiegend durch Faltungen mit Schrittweite $s = 2$ realisiert. In beiden Fällen bleibt lediglich erhalten, *ob* ein Merkmal innerhalb des Fensters auftrat, nicht seine exakte Position: Für Verschiebungen $\|\tau\|_\infty < p$ innerhalb desselben Pooling-Fensters bleibt die Ausgabe unverändert (LeCun et al., 1998; Goodfellow et al., 2016).

Die Ausgabegröße nach einer Schicht mit Kernelgröße $k$, Schrittweite $s$ und Randauffüllung $p_{\text{pad}}$ beträgt

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p_{\text{pad}} - k}{s} \right\rfloor + 1,$$

woraus für $s = 2$ näherungsweise eine Halbierung je Stufe folgt. Nach $L$ solchen Stufen beträgt die Gesamtschrittweite $S = \prod_{\ell} s_\ell = 2^L$; die Zahl der räumlichen Positionen sinkt dabei um den Faktor $S^2$, was den mit der Tiefe wachsenden Kanalzahlen entgegenwirkt.

Der Preis ist ein Verlust an Ortsinformation, der bei kleinen Objekten kritisch wird. Eine Zelle der Merkmalsebene mit Gesamtschrittweite $S$ entspricht einem Bildbereich von $S \times S$ Pixeln. Ein Objekt mit Durchmesser $d$ Pixeln belegt dort nur $d/S$ Zellen; für $d < S$ ist es kleiner als eine einzelne Zelle und kann strukturell nicht mehr zuverlässig lokalisiert werden. Konkret gilt für die feinste Detektionsebene P3 mit $S = 8$: Ein Rohrprofil mit $d = 6$ Pixeln unterschreitet diese Grenze. Dieser Zusammenhang begründet sowohl die Multi-Skalen-Fusion moderner Detektoren (Abschnitt 2.2.4) als auch die in Abschnitt 2.3.4 diskutierten Gegenmaßnahmen der erhöhten Eingabeauflösung bzw. Kachelung, die beide $d$ relativ zu $S$ vergrößern.

**Zusammenspiel.** Die drei Prinzipien reduzieren die Parameterzahl in aufeinander aufbauenden Schritten: Lokale rezeptive Felder senken die Verbindungen je Neuron von $H \cdot W$ auf $k^2$, Weight Sharing eliminiert die Abhängigkeit der Parameterzahl von der Bildgröße, und Subsampling reduziert die Zahl der zu verarbeitenden Positionen um $S^2$ und erzeugt zugleich lokale Translationsinvarianz. Gemeinsam kodieren sie genau jene Annahmen, die für Bilddaten zutreffen: Lokalität, Stationarität und Toleranz gegenüber kleinen Verschiebungen. LeCun et al. (1998) formulierten diese Prinzipien am Beispiel der Architektur LeNet-5 zur Ziffernerkennung; sie gelten in modernen Detektionsarchitekturen bis hin zu YOLO26 unverändert fort.

### 2.1.3 Aufbau eines CNNs

Ein typisches CNN besteht aus einer Abfolge folgender Schichttypen:

**Faltungsschichten (Convolutional Layers).** Kernstück des CNNs ist die diskrete Faltungsoperation. Ein Filterkern (Kernel), typischerweise der Größe 3×3 oder 5×5, wird über das Eingabebild bzw. die Merkmalskarte (Feature Map) geschoben. An jeder Position wird das Skalarprodukt zwischen Kernelgewichten und dem darunterliegenden Bildausschnitt berechnet. Das Ergebnis ist eine neue Merkmalskarte, die angibt, wie stark das vom Kernel repräsentierte Muster an der jeweiligen Position ausgeprägt ist. Die Gewichte der Kernel werden nicht manuell festgelegt, sondern im Training mittels Backpropagation gelernt (LeCun et al., 1998; Goodfellow et al., 2016).

**Aktivierungsfunktionen.** Auf jede Faltung folgt eine nichtlineare Aktivierungsfunktion, ohne die das Netz nur lineare Abbildungen lernen könnte. In modernen CNNs hat sich die Rectified Linear Unit (ReLU), f(x) = max(0, x), sowie deren Varianten (z. B. SiLU/Swish) durchgesetzt, da sie das Problem verschwindender Gradienten abmildert und effizient zu berechnen ist (Krizhevsky et al., 2012).

**Pooling-Schichten.** Pooling-Operationen (meist Max-Pooling) reduzieren die räumliche Auflösung der Merkmalskarten, indem sie z. B. aus jedem 2×2-Bereich nur den Maximalwert übernehmen. Dies verringert den Rechenaufwand, vergrößert das rezeptive Feld nachfolgender Schichten und sorgt für eine gewisse Robustheit gegenüber kleinen Verschiebungen (Goodfellow et al., 2016).

**Vollvernetzte Schichten (Fully Connected Layers).** Am Ende klassischer CNN-Architekturen fassen vollvernetzte Schichten die extrahierten Merkmale zusammen und bilden sie z. B. auf Klassenwahrscheinlichkeiten ab. In modernen Detektionsarchitekturen werden sie zunehmend durch faltungsbasierte Prädiktionsköpfe ersetzt.

### 2.1.4 Hierarchische Merkmalsextraktion

Ein wesentliches Merkmal tiefer CNNs ist der hierarchische Aufbau der gelernten Repräsentationen: Frühe Schichten reagieren auf einfache Primitive wie Kanten und Farbverläufe, mittlere Schichten kombinieren diese zu Texturen und Teilstrukturen (z. B. Rundungen, Rohrprofile), und tiefe Schichten repräsentieren komplexe, semantische Objektbestandteile (Zeiler & Fergus, 2014). Diese automatische, datengetriebene Merkmalsextraktion tritt an die Stelle des in der klassischen Bildverarbeitung manuell zu entwerfenden Feature Engineering (etwa der Hough-Transformation zur Kreiserkennung); das Lernen von Repräsentationen direkt aus Rohdaten gilt als ein wesentlicher Grund für die Leistungssprünge tiefer Netze bei Erkennungsaufgaben mit hoher Klassenzahl und großer Erscheinungsvarianz (LeCun et al., 2015), wie sie erstmals im ImageNet-Wettbewerb 2012 gegenüber merkmalsbasierten Verfahren demonstriert wurden (Krizhevsky et al., 2012). Die Überlegenheit ist allerdings an diese Bedingungen geknüpft: Bei eng umrissenen Aufgaben mit geometrisch einfachen, gut modellierbaren Objekten können klassische Verfahren vergleichbare oder bessere Ergebnisse bei erheblich geringerem Daten- und Rechenaufwand erzielen (vgl. Kapitel 3).

Den Durchbruch tiefer CNNs markierte AlexNet, das den ImageNet-Wettbewerb 2012 mit deutlichem Abstand gewann (Krizhevsky et al., 2012). Nachfolgende Architekturen wie VGG (Simonyan & Zisserman, 2015) und ResNet (He et al., 2016) vertieften die Netze weiter; insbesondere die von ResNet eingeführten Residualverbindungen (Skip Connections) ermöglichten das stabile Training sehr tiefer Netze und sind heute Standardbausteine nahezu aller CNN-Backbones – auch in der YOLO-Familie.

### 2.1.5 Objektdetektion mit CNNs

Für die vorliegende Arbeit ist nicht die reine Bildklassifikation relevant, sondern die Objektdetektion: das gleichzeitige Lokalisieren (Bounding Box) und Klassifizieren mehrerer Objekte in einem Bild. Hier haben sich zwei Paradigmen etabliert:

- **Zweistufige Detektoren** (z. B. R-CNN, Faster R-CNN) erzeugen zunächst Regionsvorschläge (Region Proposals) und klassifizieren diese anschließend (Ren et al., 2015). Sie erreichen hohe Genauigkeit, sind jedoch vergleichsweise langsam.
- **Einstufige Detektoren** (z. B. SSD, YOLO) formulieren die Detektion als einmaliges Regressionsproblem: Ein einziges Netz sagt in einem Durchlauf direkt Bounding Boxes und Klassenwahrscheinlichkeiten für das gesamte Bild vorher (Redmon et al., 2016). Dies ermöglicht Echtzeitfähigkeit bei nur geringen Genauigkeitseinbußen.

Ein klassisches Element einstufiger Detektoren ist die Non-Maximum Suppression (NMS): Da das Netz pro Objekt typischerweise mehrere überlappende Boxen vorhersagt, werden in einem Nachverarbeitungsschritt redundante Detektionen anhand ihrer Überlappung (Intersection over Union, IoU) unterdrückt. Wie in Abschnitt 2.2 gezeigt wird, ist gerade dieser Schritt bei dicht gepackten, gleichartigen Objekten – wie gestapelten Bauzäunen – eine potenzielle Fehlerquelle.

## 2.2 YOLO26

### 2.2.1 Einordnung in die YOLO-Familie

„You Only Look Once" (YOLO) wurde 2016 von Redmon et al. als erster echtzeitfähiger einstufiger Detektor vorgestellt (Redmon et al., 2016). Seither wurde die Architektur in zahlreichen Iterationen weiterentwickelt; ab YOLOv8 übernahm das Unternehmen Ultralytics die Rolle des primären Maintainers und etablierte ankerfreie Prädiktionsköpfe, entkoppelte Detektionsköpfe und eine produktionsreife Werkzeugkette (Sapkota et al., 2025). YOLO26 wurde im September 2025 auf der YOLO Vision 2025 vorgestellt und ist das aktuellste Mitglied dieser Familie. Es ist explizit auf Effizienz, Genauigkeit und einfache Bereitstellung auf Edge- und leistungsschwachen Geräten ausgelegt (Sapkota et al., 2025; Ultralytics, 2026).

### 2.2.2 Zentrale architektonische Neuerungen

YOLO26 unterscheidet sich von seinen Vorgängern (insbesondere YOLO11) durch vier koordinierte Änderungen an Architektur und Trainingsverfahren (Ultralytics, 2026; Sapkota et al., 2025):

**End-to-End-Inferenz ohne NMS.** YOLO26 verwendet ein Dual-Head-Design: Neben einem klassischen One-to-Many-Kopf, der im Training zusätzliche Supervision liefert, besitzt das Netz einen One-to-One-Prädiktionskopf, der pro Objekt genau eine Detektion ausgibt. Zur Inferenzzeit ist dieser Kopf der Standard, wodurch die NMS-Nachverarbeitung vollständig entfällt. Das reduziert die Latenz, vereinfacht den Export (ONNX, TensorRT, CoreML, TFLite, OpenVINO) und eliminiert den IoU-Schwellwert der NMS als fehleranfälligen Hyperparameter.

**Wegfall der Distribution Focal Loss (DFL).** Frühere YOLO-Versionen modellierten Boxkoordinaten als diskrete Verteilungen (DFL), was den Detektionskopf verkomplizierte und den Regressionsbereich beschränkte. YOLO26 entfernt die DFL vollständig, was einen leichteren Kopf mit unbeschränktem Regressionsbereich ergibt und den Modellexport vereinfacht.

**ProgLoss und STAL.** Die Progressive Loss (ProgLoss) verlagert die Supervision im Trainingsverlauf zunehmend auf den zur Inferenz genutzten One-to-One-Kopf. Die Small-Target-Aware Label Assignment (STAL) garantiert, dass auch sehr kleine Objekte im Training positive Labelzuweisungen erhalten – ein bekanntes Problem klassischer Zuweisungsstrategien, bei denen kleinste Objekte mitunter ohne positives Trainingssignal bleiben. Beide Maßnahmen verbessern gezielt die Erkennung kleiner Objekte.

**MuSGD-Optimierer.** Für das Training kombiniert YOLO26 den aus dem Training großer Sprachmodelle adaptierten Muon-Optimierer mit klassischem SGD (MuSGD), was eine stabilere und schnellere Konvergenz ermöglicht.

### 2.2.3 Modellvarianten und Leistung

YOLO26 ist in fünf Skalierungen verfügbar (n/s/m/l/x) und erreicht auf dem COCO-Benchmark 40,9–57,5 mAP bei Latenzen von 1,7–11,8 ms (T4-GPU, TensorRT). Die Nano-Variante ist auf CPUs bis zu 43 % schneller als YOLO11n, was den Fokus auf Edge-Deployment unterstreicht (Ultralytics, 2026). Neben der Objektdetektion unterstützt YOLO26 als Multi-Task-Framework auch Instanzsegmentierung, Posenschätzung, Klassifikation und orientierte Objektdetektion (OBB) in einer einheitlichen Pipeline (Sapkota et al., 2025). Zusätzlich existieren YAML-Konfigurationen mit einem zusätzlichen P2-Detektionskopf, der eine höher aufgelöste Merkmalsebene für sehr kleine Objekte bereitstellt (Ultralytics, 2026).

### 2.2.4 Architektur im Detail: Backbone, Neck, Head

YOLO26 folgt der für einstufige Detektoren üblichen Dreiteilung **Backbone → Neck → Head**. Das Backbone ist ein tiefes CNN, das aus dem Eingabebild (standardmäßig 640×640×3) eine Hierarchie von Merkmalskarten extrahiert; das Neck fusioniert diese Karten über mehrere Auflösungsstufen hinweg; der Head übersetzt sie in Bounding Boxes und Klassenwahrscheinlichkeiten. Das gesamte Netz besteht aus einem einzigen Vorwärtsdurchlauf – daher der Name „You Only Look Once" (Redmon et al., 2016).

**Faltungsschichten im Backbone.** Jede Faltungsschicht besteht aus lernbaren Kerneln (typisch 3×3), die über die Eingabe geschoben werden und an jeder Position das Skalarprodukt mit dem darunterliegenden Bildausschnitt bilden (vgl. Abschnitt 2.1.3). Drei Eigenschaften machen dies für die Detektion wirkungsvoll: Erstens die **Gewichtsteilung** – derselbe Kernel wird an jeder Bildposition angewendet, wodurch ein gelerntes Muster (z. B. eine helle horizontale Linie) überall im Bild mit denselben wenigen Parametern erkannt wird. Zweitens die **Lokalität**: Jedes Neuron „sieht" zunächst nur einen kleinen Bildausschnitt (sein rezeptives Feld), das mit der Netztiefe wächst – frühe Schichten erfassen Kanten, tiefe Schichten größere Objektkontexte (Zeiler & Fergus, 2014). Drittens die **Kanalstruktur**: Eine Schicht mit z. B. 64 Kerneln erzeugt 64 Merkmalskarten, von denen jede die räumliche Verteilung eines anderen gelernten Musters repräsentiert. Das Backbone reduziert die räumliche Auflösung stufenweise per Stride-2-Faltungen (640 → 320 → 160 → 80 → 40 → 20 Pixel), während die Kanalzahl wächst – räumliche Präzision wird gegen semantische Tiefe getauscht. Jeder Faltung folgen Batch-Normalisierung und eine SiLU-Aktivierung; die Blöcke sind im CSP-Stil mit Residualverbindungen organisiert (Aufteilung des Kanalstroms, parallele Verarbeitung, Wiederzusammenführung), was Gradientenfluss und Recheneffizienz verbessert (vgl. He et al., 2016). Am Ende des Backbones aggregiert ein SPPF-Modul Kontextinformation über mehrere Pooling-Größen.

**Neck: Multi-Skalen-Fusion.** Das Neck verbindet die Auflösungsstufen des Backbones in beide Richtungen (FPN/PAN-Prinzip): Tiefe, semantisch reiche Merkmalskarten werden hochskaliert und mit den feinen, ortsgenauen Karten früherer Stufen zusammengeführt, und umgekehrt. Der Head erhält dadurch drei Ausgabeebenen (P3/P4/P5, entsprechend 80×80, 40×40 und 20×20 Gitterzellen), die für kleine, mittlere und große Objekte zuständig sind. Für die Holmzählung ist insbesondere die feine P3-Ebene relevant, da die Holme innerhalb einer 640er-Kachel als vergleichsweise kleine Objekte erscheinen.

**Der Head und die Rolle der Fully-Connected-Schichten.** Historisch markiert der Prädiktionskopf den stärksten architektonischen Wandel der YOLO-Familie. Das ursprüngliche YOLOv1 (Redmon et al., 2016) endete in zwei **vollvernetzten (Fully-Connected, FC) Schichten**: Die letzte Merkmalskarte wurde zu einem eindimensionalen Vektor „plattgedrückt", durch eine FC-Schicht mit 4096 Neuronen geleitet – jedes Neuron mit jedem Eingang verbunden – und anschließend auf einen 7×7×30-Ausgabetensor abgebildet. Dies verlieh dem Netz zwar globalen Bildkontext, brachte jedoch drei Nachteile mit sich: Die FC-Schichten enthielten den überwiegenden Teil aller Netzparameter, sie fixierten die Eingabebildgröße (der plattgedrückte Vektor benötigt eine feste Länge) und sie zerstörten die räumliche Struktur der Merkmalskarten.

Moderne YOLO-Architekturen – einschließlich YOLO26 – enthalten daher **keine Fully-Connected-Schicht mehr**; sie sind vollständig faltungsbasiert (*fully convolutional*). Der Prädiktionskopf besteht aus kleinen Faltungsstapeln, die in 1×1-Faltungen münden. Mathematisch betrachtet ist eine 1×1-Faltung äquivalent zu einer Fully-Connected-Schicht über die Kanaldimension – der entscheidende Unterschied ist, dass sie **positionsgeteilt** angewendet wird: Dieselbe kleine „FC-Operation" verarbeitet die Kanäle jeder Gitterzelle separat, statt einmal global über das gesamte plattgedrückte Bild zu operieren. Jede der z. B. 80×80 Zellen der P3-Ebene sagt so lokal und unabhängig voneinander vier Boxkoordinaten sowie Klassen-Scores vorher. Dies erhält die Translationsäquivarianz der Faltung bis in die Ausgabeschicht hinein, macht das Netz unabhängig von einer festen Eingabegröße und reduziert die Parameterzahl um Größenordnungen gegenüber dem ursprünglichen FC-Ansatz – eine notwendige Voraussetzung dafür, dass die Nano-Variante YOLO26n auf mobilen Endgeräten lauffähig ist (vgl. Abschnitt 2.2.3).

Die YOLO26-spezifischen Head-Eigenschaften bauen auf dieser vollständig faltungsbasierten Struktur auf: Der Head ist *entkoppelt* (getrennte Faltungszweige für Klassifikation und Boxregression) und *ankerfrei* (jede Gitterzelle regressiert direkt Abstände zu den vier Boxrändern, statt Korrekturen vordefinierter Ankerboxen zu lernen). YOLO26 verzichtet zudem auf die in Vorgängerversionen genutzte Distribution Focal Loss; die Boxkoordinaten werden wieder als direkte Regressionswerte statt als diskrete Verteilungen vorhergesagt, was den Kopf zusätzlich verschlankt (Ultralytics, 2026). Das in Abschnitt 2.2.2 beschriebene Dual-Head-Design (One-to-Many-Zweig als zusätzliche Trainingssupervision, One-to-One-Zweig für die NMS-freie Inferenz) ist ebenfalls vollständig faltungsbasiert umgesetzt: Der One-to-One-Kopf gibt zur Laufzeit pro Objekt genau eine Vorhersage aus, wodurch die sonst nötige Non-Maximum-Suppression entfällt.

Zusammenfassend lässt sich der architektonische Kernwandel wie folgt festhalten: Faltungsschichten extrahieren im Backbone translationsinvariante, hierarchische Merkmale; die historisch nachgeschaltete Fully-Connected-Schicht ist in modernen YOLO-Architekturen vollständig durch positionsgeteilte 1×1-Faltungen ersetzt. Dieser Übergang von einer global-dichten zu einer lokal-geteilten Vorhersage ist eine wesentliche Voraussetzung für die Echtzeitfähigkeit und Edge-Tauglichkeit von YOLO26.

## 2.3 Anwendung auf die Zählung gestapelter Bauzäune

### 2.3.1 Problemstellung aus Sicht der Bildverarbeitung

Im vorliegenden Anwendungsfall sollen 50–200 aufeinandergestapelte Bauzäune gezählt werden. Die Aufnahme erfolgt frontal auf die Stirnseite des Stapels („von unten" entlang der Stapelachse), sodass pro Zaunelement im Wesentlichen die beiden runden Säulenprofile (Rohrenden) sowie der Querschnitt der verbindenden Metallstange sichtbar sind. Aus Sicht der Bildverarbeitung ergeben sich daraus folgende Charakteristika:

1. **Hohe Objektdichte:** Pro Stapelaufnahme sind etwa 50–100 Rohrprofile bzw. Holme sichtbar, dicht gepackt auf engem Raum.
2. **Kleine, gleichartige Objekte:** Die einzelnen Rohrprofile sind im Bild klein und untereinander nahezu identisch – es gibt kaum individuelle Merkmale zur Unterscheidung benachbarter Instanzen.
3. **Regelmäßige, aber gestörte Struktur:** Die Stapelung erzeugt ein quasi-periodisches Muster, das jedoch durch verschobene, verkippte oder verdeckte Elemente gestört sein kann.
4. **Perspektivische Effekte:** Je nach Kameraposition variieren Größe und Form der Profile über das Bild (nahe Elemente größer, entfernte kleiner; Kreise erscheinen als Ellipsen).

### 2.3.2 Lösungsansatz mit CNN/YOLO26

Der naheliegende Ansatz ist eine detektionsbasierte Zählung („counting by detection"): Ein YOLO26-Modell wird per Transfer Learning (Ausgangspunkt: COCO-vortrainierte Gewichte) auf eine einzige Objektklasse – das Rohrende bzw. das Säulenprofil-Paar – feinabgestimmt. Die Zaunanzahl ergibt sich dann entweder direkt (eine Detektion pro Zaun, wenn das Profilpaar als ein Objekt annotiert wird) oder als Anzahl detektierter Rohrenden geteilt durch zwei. Letzteres bietet eine eingebaute Plausibilitätsprüfung: Eine ungerade Detektionszahl bzw. eine Diskrepanz zwischen linker und rechter Profilsäule signalisiert Detektionsfehler. Da die Profile in zwei annähernd vertikalen Säulen angeordnet sind, kann die Zählung zudem säulenweise erfolgen und der Median beider Säulen als robuster Schätzer dienen.

Die hierarchische Merkmalsextraktion des CNN-Backbones (Abschnitt 2.1.4) ist für diese Aufgabe gut geeignet: Kreisförmige Metallprofile mit charakteristischem Hell-Dunkel-Verlauf (Rohrwand, dunkle Öffnung, ggf. Verzinkungsglanz) sind visuell einfache, texturarme Objekte, die bereits von mittleren Netzschichten zuverlässig repräsentiert werden können.

### 2.3.3 Stärken des Ansatzes

**Kleine-Objekt-Erkennung durch STAL und ProgLoss.** Die zentrale Herausforderung – viele kleine Rohrenden – adressiert YOLO26 konzeptionell direkt: STAL garantiert positive Labelzuweisungen auch für kleinste Objekte im Training, ProgLoss verbessert die Konsistenz des Inferenzkopfes. Für sehr hohe Stapel (kleine Profile am oberen Bildrand) kann zusätzlich die P2-Variante mit hochauflösendem Detektionskopf verwendet werden.

**NMS-freie Inferenz bei dichten Objekten.** Bei dicht gepackten, sich berührenden gleichartigen Objekten ist die klassische NMS eine systematische Fehlerquelle: Ein zu niedriger IoU-Schwellwert unterdrückt korrekte Detektionen benachbarter Rohre (Unterzählung), ein zu hoher lässt Duplikate zu (Überzählung). Der End-to-End-Ansatz von YOLO26 eliminiert diesen Schwellwert und damit eine der kritischsten Stellschrauben für Zählanwendungen.

**Geringer Klassenumfang und visuelle Einfachheit.** Ein Ein-Klassen-Problem mit geometrisch einfachen, kontrastreichen Objekten ist ein vergleichsweise „gutmütiges" Detektionsproblem. Bereits kleine Modellvarianten (YOLO26n/s) dürften ausreichen, was schnelles Training, geringe Annotationskosten pro Iteration und Echtzeitfähigkeit auf mobilen Endgeräten (z. B. Baustellen-Smartphone) ermöglicht – hierfür ist YOLO26 mit seinem Edge-Fokus und den vielfältigen Exportpfaden explizit ausgelegt.

**Interpretierbarkeit und Verifizierbarkeit.** Anders als reine Regressionsansätze liefert die detektionsbasierte Zählung pro Zaun eine sichtbare Bounding Box. Fehlzählungen sind damit visuell lokalisierbar und im industriellen Einsatz durch den Nutzer korrigierbar.

### 2.3.4 Schwächen und Grenzen

**Begrenzung der Detektionsanzahl.** Der standardmäßige One-to-One-Kopf von YOLO26 gibt maximal 300 Detektionen pro Bild aus (Ultralytics, 2026). Für den vorliegenden Anwendungsfall ist diese Grenze unkritisch: Auf den realen Aufnahmen sind pro Stapel etwa 50–100 Rohrprofile sichtbar, sodass ein deutlicher Sicherheitsabstand zur Obergrenze besteht. Relevant würde die Beschränkung erst bei Aufnahmen mehrerer Stapel in einem Bild oder bei Annotationsschemata, die jedes Einzelprofil separat erfassen. Für solche Fälle stehen als Abhilfe zur Verfügung: (a) die Annotation zusammengehöriger Profile als ein Objekt, (b) eine Erhöhung des `max_det`-Parameters oder (c) eine Kachelung des Bildes (Tiling/Slicing, z. B. mit SAHI), bei der Teilbilder separat detektiert und die Ergebnisse fusioniert werden.

**Auflösungsproblematik.** Die Standard-Eingabegröße von 640×640 Pixeln ist für die vorliegende Aufgabe knapp bemessen: Wird eine hochaufgelöste Stapelaufnahme auf 640 px herunterskaliert, schrumpfen die Rohrprofile auf wenige Pixel Durchmesser. Da die feinste Merkmalsebene P3 mit einem Stride von 8 arbeitet, belegt ein solches Objekt dort weniger als eine Gitterzelle und fällt unter die zuverlässige Detektionsgrenze. Zwei Strategien adressieren dies:

*(a) Erhöhte Eingabeauflösung.* Eine Vergrößerung auf beispielsweise `imgsz=2560` skaliert ein zuvor 6–8 px großes Profil auf 24–32 px und damit in einen gut detektierbaren Bereich. Der Rechenaufwand wächst jedoch quadratisch mit der Kantenlänge (Faktor 16 gegenüber 640 px), was eine deutliche Reduktion der Batch-Größe erzwingt. Zu beachten ist ferner, dass Trainings- und Inferenzauflösung übereinstimmen müssen, da sonst ein Skalenversatz zwischen gelernter und beobachteter Objektgröße entsteht.

*(b) Kachelung.* Alternativ wird das Bild in Kacheln der Trainingsauflösung zerlegt und jede Kachel bei nativer Bildauflösung verarbeitet. Die effektive Objektgröße entspricht dabei jener der hochauflösenden Variante, der Speicherbedarf bleibt jedoch konstant. Nachteilig ist der eingeschränkte Bildkontext pro Kachel sowie die Notwendigkeit, Detektionen aus Überlappungszonen zu fusionieren.

Beide Strategien führen zu vergleichbarer effektiver Auflösung bei unterschiedlichem Ressourcenprofil; ihr Vergleich bietet sich als Ablationsstudie an (vgl. Kapitel 4).

**Verdeckung und Mehrdeutigkeit.** Verkippte oder seitlich versetzte Zäune können benachbarte Profile teilweise verdecken; Schmutz, Rost oder Kabelbinder verändern das Erscheinungsbild. Zudem können Doppel- oder Fehldetektionen an visuell ähnlichen Strukturen (Schraubenköpfe, Rohrschellen, Löcher in Fußplatten) auftreten. Da bei einer Zählaufgabe jeder einzelne Fehler das Endergebnis direkt verfälscht, ist die Anforderung an die Detektionsgüte höher als bei typischen Detektionsanwendungen, in denen einzelne Fehldetektionen tolerierbar sind: Für eine exakte Zählung von 200 Objekten wäre eine Trefferquote von über 99,5 % nötig.

**Datenbedarf und Domänenlücke.** COCO-vortrainierte Gewichte enthalten keine Klasse „Bauzaunprofil"; das Modell muss auf einem eigens annotierten Datensatz feinabgestimmt werden. Die Annotation hunderter kleiner Boxen pro Bild ist aufwendig, und der Datensatz muss die reale Varianz (Beleuchtung, Witterung, Zaunhersteller, Kamerawinkel, Stapelhöhe) abdecken, um Überanpassung an einen einzelnen Bautyp zu vermeiden.

**Perspektive und Kalibrierung.** Da die Aufnahme von unten erfolgt, variiert die Objektgröße systematisch über das Bild. CNNs sind zwar translationsinvariant, aber nicht inhärent skaleninvariant; die Multi-Skalen-Architektur (Feature Pyramid) von YOLO26 mildert dies, ersetzt aber keine konsistente Aufnahme-Geometrie. Standardisierte Aufnahmebedingungen (fester Abstand, definierte Blickrichtung) reduzieren die Problemkomplexität erheblich.

**Alternative Ansätze als Referenz.** Für extrem dichte Szenen wird in der Literatur alternativ die dichtebasierte Zählung (Density Map Estimation / Crowd Counting) eingesetzt, bei der ein CNN eine Dichtekarte regressiert, deren Integral die Objektanzahl ergibt (Zhang et al., 2016). Dieser Ansatz skaliert besser mit der Objektanzahl, verzichtet aber auf instanzweise Lokalisierung und damit auf Verifizierbarkeit. Bei 50–200 gut separierbaren, regelmäßig angeordneten Objekten ist die detektionsbasierte Zählung mit YOLO26 der geeignetere Ausgangspunkt; die dichtebasierte Zählung bleibt als Vergleichsverfahren für die Evaluation relevant.

### 2.3.5 Zwischenfazit

CNNs liefern die methodische Grundlage, um die visuell einfachen, aber zahlreichen und kleinen Rohrprofile gestapelter Bauzäune robust zu erkennen. YOLO26 ist innerhalb der YOLO-Familie besonders geeignet, da seine Kerninnovationen – NMS-freie End-to-End-Inferenz, STAL/ProgLoss für kleine Objekte und Edge-optimierte Effizienz – genau die kritischen Eigenschaften des vorliegenden Zählproblems adressieren. Die Detektionsobergrenze des Standardkopfes stellt bei den beobachteten 50–100 Objekten pro Aufnahme kein praktisches Hindernis dar. Die wesentlichen Risiken liegen damit nicht in der Architektur, sondern in der Datenlage (Annotationsaufwand, Domänenabdeckung) sowie in der Auflösungsproblematik, die durch erhöhte Eingabeauflösung oder Kachelung adressiert werden muss.

## Literaturverzeichnis

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 25, 1097–1105.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep Learning. *Nature*, 521(7553), 436–444.
- Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 4898–4906.
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788.
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 28.
- Sapkota, R., et al. (2025). YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection. *arXiv preprint* arXiv:2509.25164.
- Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. In *International Conference on Learning Representations (ICLR)*.
- Ultralytics (2026). Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. *arXiv preprint* arXiv:2606.03748; Dokumentation: https://docs.ultralytics.com/models/yolo26
- Zeiler, M. D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. In *European Conference on Computer Vision (ECCV)*, 818–833.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding Deep Learning Requires Rethinking Generalization. In *International Conference on Learning Representations (ICLR)*. arXiv:1611.03530.
- Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-Image Crowd Counting via Multi-Column Convolutional Neural Network. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 589–597.

---

*Hinweis: Bitte prüfe die Zitierweise gegen die Vorgaben deines Lehrstuhls (APA/IEEE/Fußnoten) und verifiziere alle Quellenangaben vor Abgabe selbst. Der Text ist als Ausgangspunkt gedacht und sollte an deine konkrete Gliederung und deinen Schreibstil angepasst werden.*
