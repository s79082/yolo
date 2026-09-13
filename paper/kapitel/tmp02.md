## 2.1 Convolutional Neural Networks

Convolutional Neural Networks (CNNs) nutzen strukturelles Vorwissen über Bilddaten, indem sie lokale Verbindungen, geteilte Gewichte und räumliche Unterabtastung in ihrer Architektur verankern. Diese Eigenschaften unterscheiden sie von vollständig verbundenen Netzen und ermöglichen eine effiziente Verarbeitung räumlich strukturierter Daten (LeCun, Bottou u. a., 1998; Goodfellow u. a., 2016).

### 2.1.1 Lokale rezeptive Felder und geteilte Gewichte

Eine Faltungsschicht verbindet ein Ausgabeneuron nicht mit allen \(H\cdot W\) Eingangspixeln, sondern mit einem lokalen \(k\times k\)-Ausschnitt der Eingabe. Der gleiche Kernel wird dabei an allen räumlichen Positionen angewendet. Dadurch werden lokale Muster unabhängig von ihrer absoluten Position mit denselben Parametern verarbeitet. Die Zahl der Parameter einer Faltungsschicht hängt somit nicht von der räumlichen Größe der Eingabe ab.

Diese gemeinsame Verwendung der Gewichte entspricht der Annahme, dass ein bestimmtes lokales Muster an verschiedenen Positionen eines Bildes auf vergleichbare Weise relevant ist. Eine Faltung ist dabei translationsäquivariant: Wird die Eingabe verschoben, verschiebt sich die resultierende Merkmalskarte entsprechend. Äquivarianz bedeutet jedoch nicht Invarianz. Die genaue räumliche Position eines Merkmals bleibt in der Faltung grundsätzlich erhalten; eine gewisse Robustheit gegenüber kleinen Verschiebungen kann erst durch nachfolgende Operationen wie räumliches Subsampling entstehen.

### 2.1.2 Räumliches Subsampling

Für die vorliegende Arbeit ist insbesondere die schrittweise Reduktion der räumlichen Auflösung relevant. Sie wird klassisch beispielsweise durch Pooling und in modernen Architekturen überwiegend durch Faltungen mit Schrittweite \(s=2\) realisiert. Nach \(L\) solchen Reduktionsstufen beträgt der Gesamtschritt

$$
S=\prod_{\ell=1}^{L}s_\ell .
$$

Bei ausschließlich halbierenden Stufen gilt damit \(S=2^L\).

Durch das Subsampling wird die räumliche Auflösung der Merkmalskarten reduziert. Benachbarte Bildbereiche werden dadurch auf gröberen räumlichen Positionen repräsentiert; gleichzeitig geht ein Teil der exakten Ortsinformation verloren. Dieser Kompromiss ist für Objektdetektion grundsätzlich erwünscht, da tiefere Merkmalsebenen einen größeren Bildbereich mit zunehmend semantischer Information repräsentieren können. Für sehr kleine Objekte kann der Informationsverlust jedoch kritisch werden.

Eine Merkmalsebene mit Gesamtschrittweite \(S\) besitzt Gitterpositionen mit einem Abstand von \(S\) Bildpixeln. Ein Objekt mit charakteristischer Größe \(d\) wird auf dieser Ebene somit nur durch ungefähr \(d/S\) räumliche Positionen über seine Ausdehnung repräsentiert. Mit abnehmendem Verhältnis \(d/S\) stehen dem Detektor folglich immer weniger räumliche Positionen zur Verfügung, um das Objekt zu lokalisieren und ihm positive Trainingszuweisungen zuzuordnen. Die genaue Grenze hängt dabei vom verwendeten Zuweisungsverfahren und von der Lage des Objekts relativ zum Gitter ab.

Diese Abhängigkeit zwischen Objektgröße und Schrittweite ist ein wesentlicher Grund für die Verwendung mehrerer Detektionsebenen. Feine Ebenen bewahren mehr Ortsinformation und eignen sich daher insbesondere für kleine Objekte, während gröbere Ebenen größere räumliche Kontexte mit höherer semantischer Abstraktion bereitstellen.

### 2.1.3 Merkmalshierarchie

Die Faltungsschichten eines CNN bilden eine hierarchische Repräsentation der Eingabedaten. Nichtlineare Aktivierungsfunktionen wie die Rectified Linear Unit

$$
f(x)=\max(0,x)
$$

oder moderne Varianten wie SiLU ermöglichen dabei die Modellierung nichtlinearer Zusammenhänge. Ohne Nichtlinearitäten würde die Hintereinanderschaltung linearer Schichten wiederum eine lineare Abbildung ergeben (Krizhevsky u. a., 2012).

Mit zunehmender Netzwerktiefe verändern sich die von den Merkmalskarten repräsentierten Strukturen. Frühe Schichten reagieren typischerweise auf einfache lokale Strukturen wie Kanten und Farbverläufe, während tiefere Schichten zunehmend komplexere Texturen, Teilstrukturen und semantische Objektmerkmale repräsentieren (Zeiler und Fergus, 2014). Diese hierarchische Merkmalsextraktion ersetzt einen großen Teil des klassischen manuellen Feature Engineering und stellt einen wesentlichen Bestandteil moderner Deep-Learning-Verfahren zur Bildanalyse dar (LeCun, Bengio u. a., 2015).

Residualverbindungen ermöglichen dabei die stabile Optimierung tiefer Netzwerke, indem sie die direkte Weitergabe von Merkmalen über mehrere Schichten hinweg ermöglichen (He u. a., 2016).

Für die vorliegende Arbeit ist nicht die Bildklassifikation, sondern die Objektdetektion relevant. Ziel ist hierbei, mehrere Objekte gleichzeitig zu lokalisieren und ihren Klassen zuzuordnen. Zweistufige Detektoren erzeugen zunächst Regionsvorschläge und klassifizieren diese anschließend, während einstufige Detektoren Lokalisierung und Klassifikation in einem gemeinsamen Vorhersageschritt durchführen (Ren u. a., 2015; Redmon u. a., 2016).

Bei dicht gepackten gleichartigen Objekten stellt insbesondere die Unterdrückung redundanter Detektionen eine Herausforderung dar. Klassische Detektoren verwenden hierzu häufig Non-Maximum Suppression (NMS), bei der stark überlappende Vorhersagen anhand ihrer Intersection over Union (IoU) miteinander verglichen und unterdrückt werden. Bei dicht benachbarten Objekten kann die Wahl des NMS-Schwellenwertes dabei zu einem Zielkonflikt zwischen der Unterdrückung von Duplikaten und dem Erhalt tatsächlich unterschiedlicher Objekte führen.

## 2.2 YOLO26

### 2.2.1 Einordnung

Die YOLO-Familie („You Only Look Once“) wurde als einstufiger Echtzeit-Objektdetektor eingeführt (Redmon u. a., 2016). Die Weiterentwicklung durch Ultralytics führte unter anderem zu ankerfreien und entkoppelten Detektionsköpfen. YOLO26 wurde im September 2025 vorgestellt und kombiniert mehrere Änderungen gegenüber früheren YOLO-Architekturen, darunter ein Dual-Head-Design für NMS-freie Inferenz, der Wegfall der Distribution Focal Loss, ProgLoss in Verbindung mit Small-Target-Aware Label Assignment (STAL) sowie der MuSGD-Optimierer (Sapkota u. a., 2025; Ultralytics, 2026b).

Für die vorliegende Arbeit sind insbesondere die Änderungen relevant, die mit der Erkennung kleiner Objekte und der Verarbeitung dicht gepackter Instanzen zusammenhängen. STAL adressiert die Zuweisung von Trainingspositionen bei kleinen Objekten, während die NMS-freie Inferenz die nachgelagerte Unterdrückung überlappender Vorhersagen vermeidet.

### 2.2.2 Backbone, Neck, Head und Merkmalspyramide

YOLO26 lässt sich konzeptionell in die drei Bestandteile Backbone, Neck und Head gliedern. Das Backbone extrahiert hierarchische Merkmale und reduziert dabei schrittweise die räumliche Auflösung. Bei einer Eingabe von \(1024\times1024\) Pixeln entstehen durch Stride-2-Operationen beispielsweise Merkmalsebenen mit räumlichen Auflösungen von \(512\times512\), \(256\times256\), \(128\times128\), \(64\times64\) und \(32\times32\) Pixeln. Mit zunehmender Tiefe sinkt damit die räumliche Auflösung, während die Merkmalsdimension zunimmt.

Der Neck führt Merkmale verschiedener Auflösungsstufen wieder zusammen. Hierzu werden Prinzipien von Feature Pyramid Networks (FPN) und Path Aggregation Networks (PAN) eingesetzt (Lin, Dollár u. a., 2017). Semantisch reichhaltige Merkmale tiefer Ebenen werden auf höhere räumliche Auflösungen gebracht und mit früheren, ortsgenaueren Merkmalen kombiniert. Umgekehrt können Informationen aus feineren Ebenen wieder in tiefere Ebenen einfließen.

Der Detektionskopf erhält dadurch mehrere Merkmalsebenen, die unterschiedliche Objektgrößen abdecken. Die Standardkonfiguration verwendet die Ebenen P3, P4 und P5 mit den Gesamtschrittweiten

$$
S\in\{8,16,32\}.
$$

Für kleine Objekte kann zusätzlich eine feinere P2-Ebene mit

$$
S=4
$$

verwendet werden.

Bei einer Eingabe von \(1024\times1024\) Pixeln ergeben sich damit folgende Gittergrößen:

| Ebene | Stride \(S\) |      Gittergröße | Anzahl Positionen |
| ----- | -----------: | ---------------: | ----------------: |
| P2    |            4 | \(256\times256\) |            65 536 |
| P3    |            8 | \(128\times128\) |            16 384 |
| P4    |           16 |   \(64\times64\) |             4 096 |
| P5    |           32 |   \(32\times32\) |             1 024 |

Die P2-Ebene enthält damit allein mehr räumliche Positionen als P3, P4 und P5 zusammen. Ihre Bedeutung liegt jedoch nicht nur in der größeren Zahl von Positionen, sondern insbesondere in der geringeren Schrittweite und damit der feineren räumlichen Repräsentation kleiner Objekte.

Der Head ist vollständig faltungsbasiert. Die Klassifikation und die Boxregression werden dabei in getrennten Zweigen verarbeitet. Der Detektor ist zudem ankerfrei: Vorhersagen werden direkt für räumliche Positionen der Merkmalsebenen erzeugt, anstatt eine feste Menge vorgegebener Ankerboxen zu verwenden. Damit wird die räumliche Struktur der verwendeten Merkmalsebenen unmittelbar für die Detektion relevant.

### 2.2.3 Der P2-Detektionskopf

Alle in dieser Arbeit trainierten Modelle verwenden die Konfiguration YOLO26s-P2. Die Wahl des P2-Kopfes stellt dabei keine experimentelle Variable dar, sondern eine konstante architektonische Randbedingung. Sie wird deshalb bereits in den theoretischen Grundlagen begründet.

Die P2-Variante erweitert die Merkmalspyramide um eine zusätzliche, feinere Ebene. Während die Standardkonfiguration den Aufwärtspfad bis P3 führt und anschließend über P4 zu P5 weiterführt, wird dieser Pfad bei der P2-Variante bis zur P2-Ebene fortgesetzt. Der Detektionskopf erhält dadurch vier statt drei Eingangsmerkmalsebenen.

Die höhere räumliche Auflösung wird nicht durch ein grundsätzlich größeres Modell erkauft. Für die in dieser Arbeit verwendete Konfiguration weist die Modellzusammenfassung 9 663 464 Parameter für YOLO26s-P2 gegenüber 9 948 638 Parametern für die Basiskonfiguration aus, entsprechend einer Reduktion um etwa 2,9 %. Auch die zusätzlichen Rechenoperationen bleiben vergleichsweise begrenzt: Bei \(640\times640\) Pixeln steigt der angegebene Rechenaufwand von 22,7 auf 26,6 GFLOPs. Der wesentliche Nachteil liegt dagegen im Speicherbedarf, da die feinere P2-Merkmalskarte viermal so viele räumliche Positionen wie P3 enthält. In den eigenen Trainingsläufen stieg der Speicherbedarf bei \(imgsz=1024\) und Batchgröße 8 von etwa 6 GB auf etwa 12 GB.

Die Relevanz der zusätzlichen Ebene lässt sich über die Zahl der verfügbaren Gitterpositionen quantifizieren. Für ein kreisförmiges Objekt mit Durchmesser \(d\) und eine Merkmalsebene mit Schrittweite \(S\) beträgt die erwartete Anzahl der Gitterzentren innerhalb des Objekts

$$
E[N]
=
\frac{\pi d^2}{4S^2}.
$$

Die Herleitung und die Annahmen dieser Beziehung werden in Abschnitt 3.3.2 ausführlicher behandelt. Entscheidend ist bereits hier, dass eine Halbierung von \(S\) zu einer Vervierfachung der erwarteten Zahl solcher Positionen führt.

Für die in dieser Arbeit auftretenden Objektgrößen ist dieser Unterschied relevant. Bei einem Objekt mit Durchmesser \(11{,}6\) px beträgt die erwartete Zahl der Positionen bei \(S=8\) nur etwa 1,7, während sie bei \(S=4\) auf etwa 6,6 steigt. Im ungünstigsten Fall kann das Objekt auf der P3-Ebene sogar keine Gitterposition innerhalb seiner Ausdehnung enthalten; für P2 liegt die entsprechende geometrische Grenze deutlich niedriger.

Der P2-Kopf wird daher eingesetzt, um die architektonische Auflösungsgrenze für die kleinen Rohrprofile zu reduzieren. Er ist nicht Gegenstand einer Ablation, da andernfalls die Effekte der untersuchten Datengenerierung mit einem gleichzeitig variierenden Architekturparameter vermischt würden. Die Wahl des P2-Kopfes dient somit dazu, die räumliche Auflösung des Detektors ausreichend hoch zu halten, sodass die im weiteren Verlauf untersuchten Eigenschaften der Trainingsdaten überhaupt bewertet werden können.

## 2.3 Anwendung auf gestapelte Bauzaunprofile

Die vorliegende Aufgabe besteht in der Detektion von Rohrstirnflächen in gestapelten Bauzaunprofilen. Ein YOLO-Modell wird hierzu mittels Transfer Learning von vortrainierten Gewichten auf eine einzelne Objektklasse feinabgestimmt (Yosinski u. a., 2014).

Die Aufgabe unterscheidet sich von vielen klassischen Objektdetektionsproblemen insbesondere durch drei Eigenschaften: Die Zielobjekte sind klein, sie treten in hoher räumlicher Dichte auf und sie können sich gegenseitig verdecken. Diese Eigenschaften stellen unterschiedliche Anforderungen an Architektur, Inferenz und Trainingsdaten.

### Kleine Objekte

Die Rohrstirnflächen besitzen im untersuchten Datensatz nur wenige Pixel Durchmesser. Damit wird die effektive Objektgröße nach der Skalierung auf die Netzeingabe zu einem zentralen Parameter. Für ein Bild mit längster Kante \(N\), eine Netzeingabe mit Kantenlänge \(M\) und ein Objekt mit ursprünglichem Durchmesser \(d\) ergibt sich näherungsweise

$$
d_{\mathrm{eff}}
=
d\frac{M}{N}.
$$

Je kleiner \(d_{\mathrm{eff}}\) im Verhältnis zur Schrittweite der feinsten Detektionsebene wird, desto weniger räumliche Positionen stehen für die Repräsentation und Zuweisung des Objekts zur Verfügung. Eine höhere Eingabeauflösung erhöht \(d_{\mathrm{eff}}\), während ein zusätzlicher P2-Kopf die Schrittweite von \(S=8\) auf \(S=4\) reduziert.

Die Wahl von P2 und der später untersuchten Inferenzauflösung folgt damit demselben geometrischen Grundprinzip: Das Verhältnis zwischen effektiver Objektgröße und Feature-Stride soll ausreichend groß sein.

### Hohe Objektdichte und NMS

Die Rohrprofile liegen dicht nebeneinander, sodass sich ihre Bounding Boxes stark überlappen können. Bei klassischer NMS kann dadurch ein Konflikt entstehen: Ein zu niedriger IoU-Schwellenwert kann benachbarte, tatsächlich unterschiedliche Rohrprofile unterdrücken, während ein zu hoher Schwellenwert redundante Vorhersagen bestehen lässt.

Die NMS-freie Inferenz von YOLO26 vermeidet diese nachgelagerte Entscheidung. Damit wird ein für die vorliegende Aufgabe relevanter Einflussfaktor reduziert. Die NMS-Freiheit beseitigt jedoch nicht die eigentliche Schwierigkeit der Aufgabe: Das Modell muss weiterhin mehrere räumlich eng benachbarte und teilweise überdeckte Instanzen unterscheiden können.

### Verdeckung

Neben der geringen Objektgröße stellt die gegenseitige Verdeckung der Rohrprofile eine zentrale Herausforderung dar. Verkippte Zäune können benachbarte Stirnflächen teilweise verdecken. Zusätzlich können Gitterknoten, Öffnungen des Maschendrahts und Schattenspalten Bildstrukturen erzeugen, die einer Rohrstirnfläche ähneln.

Die Verdeckung ist deshalb nicht lediglich eine künstliche Störung zur Regularisierung des Netzes, sondern ein regulärer Bestandteil der Zieldomäne. Die Trainingsdaten müssen diese Situation daher in einer für die reale Szene plausiblen Form abbilden. Insbesondere ist zwischen der Menge der Verdeckung und ihrer visuellen Realistik zu unterscheiden.

### Konsequenzen für den Aufbau der Untersuchung

Aus den Eigenschaften der Aufgabe ergeben sich drei voneinander unterscheidbare Einflussbereiche, die im praktischen Teil getrennt betrachtet werden:

1. **Räumliche Auflösung:** Die geringe Objektgröße wird durch den P2-Detektionskopf und die Wahl der Inferenzauflösung adressiert.
2. **Verdeckung und Bildinteraktion:** Die Wirkung unterschiedlicher Verdeckungsmodelle wird experimentell untersucht.
3. **Trainingsdaten:** Da eine ausreichende Menge handannotierter Realbilder nicht zur Verfügung steht, wird der Trainingsdatensatz durch eine synthetische Datengenerierung erzeugt.

Die Architektur wird dabei konstant gehalten, damit die späteren Ablationen primär den Einfluss der Trainingsdaten untersuchen. Umgekehrt muss die Datengenerierung die räumlichen Grenzen der gewählten Architektur berücksichtigen: Eine Verbesserung der Trainingsdaten kann eine unzureichende räumliche Auflösung nicht vollständig kompensieren. Die theoretische Betrachtung von Objektgröße und Feature-Stride bildet daher die Grundlage für die in Kapitel 3 formulierten Anforderungen an die Datengenerierung.
