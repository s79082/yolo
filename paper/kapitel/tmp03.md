# 3 Datenaugmentierung und synthetische Datengenerierung

Datenaugmentierung bezeichnet die Erzeugung zusätzlicher Trainingsbeispiele aus vorhandenen Daten durch Transformationen, die die für die Lernaufgabe relevante Beschriftung erhalten (Shorten und Khoshgoftaar, 2019). Ziel ist es, die Verteilung der im Training beobachteten Beispiele zu erweitern und das Modell gegenüber Variationen robust zu machen, die auch bei der späteren Anwendung auftreten können.

Im vorliegenden Anwendungsfall wird dieser Begriff weiter gefasst. Die Trainingsdaten werden nicht lediglich durch Transformationen real aufgenommener Bilder vergrößert. Stattdessen werden die Trainingsbilder überwiegend synthetisch aus wenigen realen Ausgangsobjekten und Hintergründen erzeugt. Die zentrale Frage ist daher nicht nur, welche Transformationen auf vorhandene Bilder angewendet werden sollen, sondern welche Verteilung synthetischer Szenen erzeugt werden muss, damit sie die für die Inferenz relevante Domäne hinreichend abbildet.

## 3.1 Begriff, Formalisierung und Wirkmechanismen

Sei

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
$$

ein Trainingsdatensatz mit Eingabebildern \(x_i\) und zugehörigen Beschriftungen \(y_i\). Bei der Objektdetektion besteht \(y_i\) aus einer Menge von Objektinstanzen, beispielsweise Bounding Boxes mit Klasseninformationen.

Sei weiterhin \(T\) eine Menge zulässiger Transformationen und \(p(t)\) eine Wahrscheinlichkeitsverteilung über diese Transformationen. Für eine Transformation \(t\in T\) bezeichnet

$$
\tau_t
$$

die zugehörige Transformation der Beschriftung. Das durch Augmentierung entstehende empirische Risiko lässt sich dann schreiben als

$$
\hat R_{\mathrm{aug}}(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb E_{t\sim p(t)}
\left[
L\left(f(t(x_i)),\tau_t(y_i)\right)
\right].
\tag{3.1}
$$

Zum Vergleich ist das gewöhnliche empirische Risiko

$$
\hat R(f)
=
\frac{1}{n}
\sum_{i=1}^{n}
L\left(f(x_i),y_i\right).
\tag{3.2}
$$

Der wesentliche Unterschied besteht im Erwartungswert über \(t\). Während das empirische Risiko jedes Trainingsbeispiel in seiner ursprünglichen Form bewertet, berücksichtigt das augmentierte Risiko eine Verteilung von Varianten desselben Beispiels. Diese Sichtweise steht in Verbindung mit der Vicinal Risk Minimization (VRM), bei der nicht nur die empirischen Trainingspunkte selbst, sondern Umgebungen dieser Punkte zur Schätzung des Risikos herangezogen werden (Chapelle u. a., 2001).

### Transformation, Transformationsverteilung und Labeltransformation

Die vier Größen in Gleichung (3.1) besitzen für die vorliegende Anwendung unterschiedliche Bedeutungen.

$$
(x_i,y_i)
$$

bezeichnet das ursprüngliche Trainingsbeispiel. Bei der Objektdetektion ist \(y_i\) eine Menge von Boxen und Klassen und damit wesentlich komplexer als ein einzelnes Klassifikationslabel.

\(T\) beschreibt die Menge der prinzipiell zugelassenen Transformationen. Dazu gehören beispielsweise geometrische Transformationen wie Translation, Skalierung und Rotation sowie photometrische Transformationen wie Helligkeits-, Kontrast- oder Rauschänderungen.

Die Verteilung

$$
p(t)
$$

bestimmt, wie häufig und in welcher Stärke die einzelnen Transformationen auftreten. Sie stellt damit nicht lediglich eine technische Implementierungsentscheidung dar, sondern eine Modellierung der erwarteten Variation der Zieldomäne.

Schließlich beschreibt

$$
\tau_t
$$

die notwendige Transformation des Labels. Bei einer Helligkeitsänderung bleibt die Bounding Box beispielsweise unverändert. Bei einer Translation muss ihre Position dagegen entsprechend verschoben werden. Bei einer Rotation muss berücksichtigt werden, dass die transformierte Objektgeometrie anschließend gegebenenfalls wieder durch eine achsparallele Bounding Box approximiert wird.

Für eine Transformation, deren Einfluss auf die Beschriftung nicht exakt bekannt ist, kann dadurch zusätzliches Labelrauschen entstehen. Gerade bei kleinen Objekten ist dieser Effekt relevant, da bereits geringe geometrische Abweichungen einen vergleichsweise großen Anteil der Objektfläche betreffen können.

### Monte-Carlo-Interpretation

Da \(T\) im Allgemeinen kontinuierlich ist, kann der Erwartungswert in Gleichung (3.1) nicht vollständig ausgewertet werden. In der praktischen Umsetzung wird deshalb eine endliche Anzahl von Transformationen aus \(p(t)\) gezogen. Bei einer einzelnen Ziehung pro Bild und Epoche ergibt sich beispielsweise die Monte-Carlo-Schätzung

$$
\hat R_{\mathrm{aug}}
\approx
\frac{1}{n}
\sum_{i=1}^{n}
L\left(
f(t_i(x_i)),
\tau_{t_i}(y_i)
\right),
\qquad
t_i\sim p(t).
\tag{3.3}
$$

Ein Trainingsdatensatz kann dadurch über viele Epochen unterschiedliche Realisierungen derselben Ausgangsbeispiele erzeugen. Die Anzahl physisch gespeicherter Bilder ist daher nicht identisch mit der Zahl der vom Modell während des Trainings gesehenen Varianten.

Im vorliegenden Fall wird diese Unterscheidung besonders relevant, weil die Trainingsbilder nicht ausschließlich aus aufgenommenen Realbildern bestehen. Die Variation der synthetischen Daten wird wesentlich durch die Verteilung \(p(t)\) des Generators bestimmt.

### Augmentierung als Modellierung von Invarianzen

Eine wichtige Funktion von Datenaugmentierung besteht darin, dem Modell Variationen zu zeigen, unter denen die Zielentscheidung invariant bleiben soll. Wird beispielsweise ein Rohrende horizontal gespiegelt und bleibt die Klassenzugehörigkeit unverändert, kann das Training dem Modell diese Variation als irrelevant nahelegen.

Dabei muss jedoch zwischen einer tatsächlichen Invarianz der Zieldomäne und einer lediglich künstlich angenommenen Invarianz unterschieden werden. Wird eine Transformation auf Trainingsdaten angewendet, obwohl sie in der realen Domäne nicht labelerhaltend ist, kann dadurch diskriminative Information entfernt werden.

Daraus folgt für die Gestaltung der Pipeline ein grundlegendes Prinzip:

> Eine Transformation sollte nur in dem Maß als labelerhaltend behandelt werden, in dem die entsprechende Variation auch in der Zieldomäne auftreten kann.

Für die Rohrstirnflächen ist beispielsweise eine große Variation der Orientierung plausibel, während für andere Objekterkennungsszenarien eine vollständige Rotationsinvarianz nicht notwendigerweise sinnvoll wäre.

### Wirkmechanismen

Die Wirkung von Datenaugmentierung lässt sich vereinfacht über drei miteinander verbundene Mechanismen beschreiben.

Erstens kann Augmentierung gewünschte Invarianzen in den Trainingsdaten repräsentieren. Das Modell wird dadurch mit mehreren Darstellungen desselben semantischen Inhalts konfrontiert und soll für diese Darstellungen dieselbe Vorhersage liefern.

Zweitens können zufällige photometrische oder sensorische Störungen als Form einer Regularisierung wirken. Rauschen, Unschärfe oder Beleuchtungsvariation verhindern dabei, dass das Modell ausschließlich auf einzelne, exakt reproduzierbare Bildmerkmale zurückgreift (Bishop, 1995).

Drittens kann Augmentierung den Bereich des Eingaberaums erweitern, in dem Trainingsbeispiele vorliegen. Dies ist insbesondere bei kleinen Ausgangsdatensätzen relevant. Die zusätzlichen Beispiele stellen jedoch keine unabhängigen Beobachtungen der Objektklasse dar. Ihre Information stammt weiterhin aus den ursprünglichen Daten und den Annahmen des Generators.

Gerade bei einer überwiegend synthetischen Trainingsmenge verschiebt sich damit die Bedeutung von \(p(t)\): Die Transformationsverteilung bestimmt einen wesentlichen Teil der beobachteten Datenverteilung.

---

## 3.2 Verfahrensüberblick

Datenaugmentierung umfasst eine große Anzahl unterschiedlicher Verfahren. Für die vorliegende Arbeit sind insbesondere geometrische Transformationen, photometrische Transformationen, Verdeckungsstrategien und kompositionelle Verfahren relevant.

Geometrische Transformationen verändern die Lage oder Form von Objekten. Dazu zählen insbesondere Translation, Skalierung, Spiegelung, Rotation, Scherung und Perspektivtransformation. Photometrische Transformationen verändern dagegen die Bildintensitäten, ohne die Objektgeometrie grundsätzlich zu verändern. Beispiele sind Helligkeits- und Kontraständerungen, Farbverschiebungen, Rauschen, Unschärfe und Kompression.

Eine weitere Klasse bilden Verfahren zur künstlichen Verdeckung von Bildbereichen. Cutout, Random Erasing und Hide-and-Seek erzeugen beispielsweise teilweise verdeckte Eingaben, um das Modell gegenüber fehlender Bildinformation robuster zu machen. Für die vorliegende Anwendung ist jedoch entscheidend, dass Verdeckung kein rein künstliches Störphänomen ist. Die Zielobjekte werden in der realen Szene tatsächlich teilweise durch andere Profile, Gitterstrukturen und Schatten verdeckt.

Daher wird Verdeckung nicht primär durch zufälliges Löschen rechteckiger Bildbereiche modelliert. Stattdessen werden physikalisch plausible Objekte als Okkluder in die Szene eingefügt. Dadurch können gleichzeitig die Geometrie, die sichtbaren Objektbereiche und die Interaktion zwischen Vorder- und Hintergrund kontrolliert werden.

### Klassische Augmentierung und synthetische Datengenerierung

Bei klassischer Datenaugmentierung bildet ein reales Trainingsbild den Ausgangspunkt. Transformationen erzeugen daraus weitere Varianten:

$$
x_i
\rightarrow
t(x_i).
$$

Die Information über die reale Bilddomäne stammt somit weiterhin aus dem ursprünglichen Bild.

Bei der in dieser Arbeit verwendeten synthetischen Datengenerierung wird dagegen eine Szene aus mehreren Komponenten zusammengesetzt:

$$
x
=
G(o_1,\ldots,o_m,b,t),
\tag{3.4}
$$

wobei \(o_j\) für ein Objekt, \(b\) für einen Hintergrund und \(t\) für die übrigen Parameter des Generators steht. Der Generator \(G\) erzeugt aus diesen Komponenten ein vollständiges Trainingsbild einschließlich der zugehörigen Labels.

Dadurch wird aus der Augmentierung im engeren Sinne eine Form der prozeduralen Datengenerierung. Die Vielfalt der Trainingsdaten wird nicht mehr primär durch eine große Menge realer Ausgangsbilder bestimmt, sondern durch die Parameterisierung des Generators.

### Warum keine automatisierte Policy-Suche?

Verfahren wie AutoAugment optimieren eine Augmentierungspolicy anhand einer Validierungsleistung (Cubuk, Zoph, Mané u. a., 2019). RandAugment reduziert die Suche auf wenige globale Parameter (Cubuk, Zoph, Shlens u. a., 2020). Solche Verfahren setzen jedoch einen geeigneten Bewertungsdatensatz voraus.

Für die vorliegende Untersuchung ist eine automatische Suche über Augmentierungsparameter nur eingeschränkt sinnvoll, da die Zahl realer Validierungsbilder sehr gering ist. Eine Optimierung der Pipeline auf diesem Split würde die Gefahr erhöhen, dass die Augmentierungsparameter auf genau diese wenigen Bilder angepasst werden.

Aus diesem Grund werden die wesentlichen Parameter der Datengenerierung nicht durch eine automatische Policy-Suche optimiert, sondern aus den Anforderungen der Zieldomäne abgeleitet und anschließend in gezielten Experimenten untersucht.

### Warum keine generative Modellierung der Objekte?

Generative Modelle wie Diffusionsmodelle bieten grundsätzlich die Möglichkeit, neue Bildinhalte zu erzeugen. Für die vorliegende Anwendung wäre jedoch eine ausreichend große und repräsentative Menge realer Beispiele der Zielobjekte erforderlich, um die gewünschte Zielverteilung zu modellieren.

Da die Zahl der verfügbaren realen Ausgangsobjekte sehr klein ist, wird stattdessen ein explizit kontrollierbarer Generator verwendet. Dieser ermöglicht es, geometrische und photometrische Eigenschaften unabhängig voneinander zu variieren und die resultierenden Labels direkt aus der bekannten Szenengeometrie abzuleiten.

Die Entscheidung gegen generative Bildmodelle ist somit keine grundsätzliche Aussage über deren Leistungsfähigkeit, sondern folgt aus den Anforderungen an Kontrolle, Labelgenauigkeit und Nachvollziehbarkeit der vorliegenden Untersuchung.

---

## 3.3 Besonderheiten bei der Objektdetektion

Datenaugmentierung für Objektdetektion unterscheidet sich von der Augmentierung bei Klassifikationsaufgaben insbesondere dadurch, dass eine Transformation nicht nur das Bild, sondern auch die räumlichen Beschriftungen verändern muss.

Bei der Klassifikation kann für viele labelerhaltende Transformationen

$$
\tau_t(y)=y
$$

angenommen werden. Bei der Detektion gilt dies im Allgemeinen nicht. Wird ein Objekt beispielsweise um \((\Delta x,\Delta y)\) verschoben, müssen auch die Boxkoordinaten entsprechend verschoben werden.

Für die vorliegende Aufgabe entstehen daraus zwei besondere Anforderungen. Erstens müssen geometrische Transformationen möglichst exakt auf die Beschriftungen übertragen werden. Zweitens muss die erzeugte Objektgröße mit der räumlichen Auflösung des verwendeten Detektors vereinbar sein.

### 3.3.1 Achsparallele Bounding Boxes und Rotation

Die Zielobjekte werden mit achsparallelen Bounding Boxes beschrieben. Diese Darstellung ist für die verwendete Detektionsaufgabe zweckmäßig, führt jedoch bei geometrischen Transformationen zu einer zusätzlichen Approximation.

Betrachtet werde zunächst ein Quadrat mit Seitenlänge \(d\), das um einen Winkel \(\theta\) rotiert wird. Die Seitenlänge der kleinsten achsparallelen Bounding Box des rotierten Quadrats beträgt

$$
d_{\mathrm{box}}
=
d\left(
|\cos\theta|+|\sin\theta|
\right).
\tag{3.5}
$$

Die Fläche des ursprünglichen Quadrats beträgt

$$
A=d^2,
$$

während die Fläche der achsparallelen Hülle

$$
A_{\mathrm{box}}
=
d^2
\left(
|\cos\theta|+|\sin\theta|
\right)^2
$$

beträgt.

Wenn die ursprüngliche Box anschließend als Label der rotierten Instanz verwendet wird, ergibt sich unter dieser idealisierten Betrachtung die IoU

$$
\operatorname{IoU}(\theta)
=
\frac{A}{A_{\mathrm{box}}}
=
\frac{1}
{\left(
|\cos\theta|+|\sin\theta|
\right)^2}.
\tag{3.6}
$$

Mit

$$
(\cos\theta+\sin\theta)^2
=
1+\sin(2\theta)
$$

folgt für

$$
0\leq\theta\leq45^\circ
$$

näherungsweise

$$
\operatorname{IoU}(\theta)
=
\frac{1}{1+\sin(2\theta)}.
\tag{3.7}
$$

Für \(2^\circ\), \(10^\circ\) und \(45^\circ\) ergeben sich damit ungefähr

$$
\operatorname{IoU}(2^\circ)\approx0{,}935,
$$

$$
\operatorname{IoU}(10^\circ)\approx0{,}745,
$$

und

$$
\operatorname{IoU}(45^\circ)=0{,}5.
$$

Die Rechnung stellt einen idealisierten geometrischen Fall dar. Bei kreisförmigen Rohrstirnflächen ist die tatsächliche Objektgeometrie rotationssymmetrisch und verändert sich durch eine reine Rotation nicht. Entscheidend ist vielmehr, ob die verwendete Labeltransformation die Box aus der tatsächlich transformierten Objektgeometrie neu bestimmt oder lediglich die ursprüngliche achsparallele Box geometrisch transformiert und anschließend wieder achsparallel umschließt.

Für die vorliegende synthetische Datengenerierung kann diese Unterscheidung genutzt werden. Werden einzelne Rohrprofile vor der Komposition rotiert, kann ihre Bounding Box anschließend direkt aus der bekannten Objektmaske bestimmt werden. Eine geometrische Ungenauigkeit durch die Transformation einer bereits vorhandenen Box lässt sich dadurch vermeiden.

Aus diesem Grund werden geometrische Variationen, die im Generator label-exakt umgesetzt werden können, bevorzugt dort erzeugt und nicht zusätzlich als bildweite Trainingszeit-Transformation angewendet.

---

### 3.3.2 Kleine Objekte, Auflösung und Labelzuweisung

Die Größe eines Objekts im Eingabebild bestimmt gemeinsam mit der Schrittweite des Detektors, wie viele räumliche Positionen auf einer Merkmalsebene zur Repräsentation des Objekts zur Verfügung stehen.

Sei \(d\) der Durchmesser eines Objekts im Originalbild, \(N\) die Kantenlänge des Originalbildes und \(M\) die Kantenlänge der Netzeingabe. Unter der Annahme einer isotropen Skalierung beträgt die effektive Objektgröße

$$
d_{\mathrm{eff}}
=
d\frac{M}{N}.
\tag{3.8}
$$

Für eine Merkmalsebene mit Stride \(S\) beträgt der Abstand benachbarter Gitterpositionen \(S\). Damit ist das Verhältnis

$$
\frac{d_{\mathrm{eff}}}{S}
\tag{3.9}
$$

ein unmittelbares Maß dafür, wie fein das Objekt durch diese Ebene räumlich abgetastet wird.

Eine genauere geometrische Betrachtung kann für ein kreisförmiges Objekt durchgeführt werden. Unter der idealisierten Annahme eines regelmäßigen quadratischen Gitters beträgt die erwartete Zahl der Gitterzentren innerhalb eines Kreises mit Durchmesser \(d\)

$$
E[N]
=
\frac{\pi(d/2)^2}{S^2}
=
\frac{\pi d^2}{4S^2}.
\tag{3.10}
$$

Damit steigt die erwartete Zahl räumlicher Positionen quadratisch mit der Objektgröße und fällt quadratisch mit der Schrittweite.

Eine Halbierung der Schrittweite von

$$
S=8
$$

auf

$$
S=4
$$

erhöht die erwartete Zahl von Gitterpositionen daher um den Faktor vier.

Dies erklärt die Bedeutung einer zusätzlichen P2-Ebene für die vorliegende Aufgabe. Die P2-Ebene besitzt mit \(S=4\) eine doppelt so hohe räumliche Auflösung wie P3 mit \(S=8\). Nach der Dokumentation von Ultralytics ist P2 explizit als Small-Object-Detection-Head vorgesehen; YOLO26 kombiniert diese architektonische Möglichkeit zusätzlich mit STAL, das die positive Label-Abdeckung kleiner Objekte verbessern soll.

Die Geometrie des Gitters zeigt außerdem, dass die Existenz einer positiven Position nicht allein von der Objektgröße abhängt, sondern auch von der relativen Lage des Objekts zum Gitter. Ein Objekt kann daher selbst dann ungünstig repräsentiert sein, wenn seine Ausdehnung in der Größenordnung des Strides liegt.

Für ein Quadrat der Seitenlänge \(d\) kann beispielsweise der Abstand seines Zentrums zum nächstgelegenen Gitterzentrum maximal \(S/\sqrt2\) betragen. Damit kann es in einer idealisierten Betrachtung bereits dann vorkommen, dass kein Gitterzentrum innerhalb des Quadrats liegt, wenn

$$
d<S\sqrt2.
\tag{3.11}
$$

Diese Beziehung stellt keine allgemeine Detektionsgrenze dar. Sie beschreibt lediglich die geometrische Möglichkeit einer fehlenden Gitterposition unter den genannten Annahmen. Ob ein reales YOLO-Modell ein solches Objekt tatsächlich nicht detektieren kann, hängt zusätzlich von Feature-Extraktion, Label Assignment, Regression und Inferenz ab.

Für die vorliegende Untersuchung folgt daraus eine praktische Konsequenz: Die Trainingsdaten dürfen nicht systematisch Objektgrößen erzeugen, die im Verhältnis zur gewählten Eingabeauflösung und zum feinsten Detektions-Stride nur unzureichend repräsentiert werden.

Die Größenverteilung der Trainingsobjekte sollte daher die für die Inferenz relevante Größenverteilung hinreichend abdecken. Dies betrifft sowohl die Generierung der Objekte als auch die Wahl der Bildauflösung und der Skalierungsparameter.

---

### 3.3.3 Mehrbildverfahren und Mosaic

Verfahren wie Mosaic kombinieren mehrere Trainingsbilder zu einer gemeinsamen Eingabe. Dadurch können unterschiedliche Szenen gleichzeitig in einem Trainingsbeispiel repräsentiert werden und die Varianz der Bildkomposition erhöht werden.

Für die vorliegende Anwendung ist Mosaic jedoch ambivalent. Einerseits kann das Verfahren unterschiedliche Hintergründe, Objektpositionen und Objektanordnungen kombinieren. Andererseits verändert die Zusammensetzung mehrerer Bilder die effektive Objektgröße. Ein Objekt, das vor der Mosaic-Komposition eine bestimmte Pixelgröße besitzt, kann durch die anschließende Skalierung deutlich kleiner werden.

Damit besteht insbesondere bei kleinen Zielobjekten ein Zielkonflikt:

$$
\text{mehr Szenenvielfalt}
\quad\leftrightarrow\quad
\text{geringere effektive Objektgröße}.
$$

Diese Wechselwirkung ist für die vorliegende Untersuchung relevant, da bereits die ursprüngliche Aufgabe durch sehr kleine Rohrprofile gekennzeichnet ist. Mosaic wird daher nicht isoliert als grundsätzlich vorteilhaft oder nachteilig betrachtet, sondern im Zusammenhang mit der resultierenden Objektgrößenverteilung.

Auch die offizielle YOLO26-Trainingskonfiguration verwendet starke Mosaic-Augmentierung, schaltet sie jedoch gegen Ende des Trainings ab. Gleichzeitig wird eine starke Skalierungsaugmentation eingesetzt, während Rotation und Scherung vergleichsweise gering gehalten werden.

---

## 3.4 Rein synthetische Trainingsdaten und Sim-to-Real

Wenn nahezu der gesamte Trainingsdatensatz synthetisch erzeugt wird, verschiebt sich die zentrale Fragestellung von der klassischen Datenaugmentation zur Modellierung einer Trainingsdatenverteilung.

Sei

$$
D_S
$$

die synthetische Trainingsdomäne und

$$
D_T
$$

die reale Zieldomäne. Das trainierte Modell minimiert zunächst einen Fehler auf \(D_S\), soll aber im späteren Einsatz auf \(D_T\) gute Vorhersagen erzeugen.

Eine allgemeine Betrachtung des Domain Gap lässt sich über die bekannte Domänenadaptationsschranke von Ben-David u. a. formulieren:

$$
\epsilon_T(h)
\leq
\epsilon_S(h)
+
\frac{1}{2}
d_{\mathcal H\Delta\mathcal H}(D_S,D_T)
+
\lambda.
\tag{3.12}
$$

Dabei bezeichnet \(\epsilon_S(h)\) den Fehler auf der Quelldomäne, \(\epsilon_T(h)\) den Fehler auf der Zieldomäne, \(d_{\mathcal H\Delta\mathcal H}\) eine Distanz zwischen den Domänen und \(\lambda\) den gemeinsamen Fehler eines optimalen Modells über beide Domänen.

Die Gleichung macht drei Einflussgrößen sichtbar:

1. Wie gut das Modell die synthetische Domäne lernt,
2. wie stark sich synthetische und reale Domäne unterscheiden,
3. ob eine Hypothese existiert, die beide Domänen gleichzeitig gut beschreibt.

Für die vorliegende Arbeit wird diese Schranke nicht zur direkten Schätzung der Domänendistanz verwendet. Der beobachtete Unterschied

$$
\Delta_{\mathrm{SR}}
=
\epsilon_T(h)-\epsilon_S(h)
\tag{3.13}
$$

wird stattdessen als **Sim-to-Real-Performance-Gap** bezeichnet. Aus Gleichung (3.12) folgt lediglich

$$
\Delta_{\mathrm{SR}}
\leq
\frac12d_{\mathcal H\Delta\mathcal H}(D_S,D_T)+\lambda,
\tag{3.14}
$$

sofern die Voraussetzungen der zugrunde liegenden Schranke erfüllt sind. Ein beobachteter Performance-Gap misst damit nicht unmittelbar die Distribution Divergence. Er kann jedoch als empirischer Indikator dafür dienen, dass die synthetische Trainingsdomäne nicht vollständig mit der realen Zieldomäne übereinstimmt.

### Domain Randomization

Eine Möglichkeit, den Sim-to-Real-Gap zu reduzieren, besteht darin, die synthetische Domäne absichtlich über die erwartete reale Variation hinaus zu erweitern. Bei Domain Randomization werden Parameter des Generators zufällig variiert, sodass das Modell nicht auf eine einzige, eng definierte Erscheinungsform angewiesen ist (Tobin u. a., 2017).

Für die vorliegende Arbeit bedeutet dies beispielsweise Variation von:

* Objektposition und Orientierung,
* Objektgröße,
* Hintergrund,
* Helligkeit und Kontrast,
* Rauschen und Unschärfe,
* Kompression,
* Verdeckung.

Domain Randomization löst jedoch nicht automatisch das Problem systematischer Unterschiede zwischen synthetischen und realen Bildern. Wenn eine bestimmte Eigenschaft in sämtlichen synthetischen Bildern auftritt, bleibt sie trotz zufälliger Variation anderer Parameter eine mögliche Abkürzung für das Modell.

### Shortcut Learning

Ein Modell kann für eine Aufgabe relevante Vorhersagen über Merkmale treffen, die zwar im Trainingsdatensatz korreliert, aber nicht kausal oder stabil mit der Zielklasse verbunden sind. Dieses Verhalten wird als Shortcut Learning bezeichnet (Geirhos u. a., 2020).

Für eine synthetische Datengenerierung entsteht ein besonderes Risiko, da technische Eigenschaften des Generators systematisch wiederkehren können. Beispielsweise können harte Objektkanten, identische Schattenmuster oder charakteristische Farbübergänge in allen synthetischen Bildern auftreten. Ein Modell kann solche Merkmale nutzen, obwohl sie in realen Bildern nicht vorhanden sind.

Dabei muss zwischen zufälliger Variation und systematischem Bias unterschieden werden. Zufällige Variation kann die Vielfalt der beobachteten Beispiele erhöhen. Ein systematischer Fehler, der in allen synthetischen Bildern vorhanden ist, wird dagegen durch zusätzliche Ziehungen aus derselben fehlerhaften Verteilung nicht beseitigt.

Dies ist für die vorliegende Arbeit besonders relevant: Die Qualität der synthetischen Daten hängt nicht nur davon ab, wie viele verschiedene Bilder erzeugt werden, sondern auch davon, welche konstanten visuellen Signaturen der Generator selbst einführt.

### Potenzielle synthetische Signaturen

Für die verwendete Pipeline werden sechs potenzielle Quellen systematischer Unterschiede zwischen synthetischen und realen Bildern betrachtet:

| Signatur | Ursache                                       | Mögliche Gegenmaßnahme                           |
| -------- | --------------------------------------------- | ------------------------------------------------ |
| S1       | sichtbare Übergänge an eingefügten Objekten   | weiche Masken, Farbangleichung                   |
| S2       | künstliche Helligkeits- oder Farbverhältnisse | Variation und Matching photometrischer Parameter |
| S3       | unrealistische Schärfe oder Auflösung         | Blur- und Downsampling-Variation                 |
| S4       | fehlendes bzw. unrealistisches Sensorrauschen | Rauschmodell                                     |
| S5       | fehlende Kompressionsartefakte                | JPEG-Kompression                                 |
| S6       | fehlende Interaktion mit der Umgebung         | physikalisch plausible Okkluder und Schatten     |

Die Tabelle stellt keine vollständige Beschreibung aller möglichen Domain-Gap-Ursachen dar. Sie definiert vielmehr die sechs Signaturen, die für die in dieser Arbeit entwickelte Pipeline als besonders relevant betrachtet und experimentell untersucht werden.

### Verdeckung als Domäneneigenschaft

Bei den untersuchten Rohrprofilen ist Verdeckung ein normaler Bestandteil der Zielszene. Ein Rohrende kann teilweise durch davorliegende Profile, das Gitter oder andere Bildelemente verdeckt werden.

Daraus ergibt sich ein Unterschied zu klassischen Verfahren wie Cutout oder Random Erasing. Dort wird Verdeckung primär als künstliche Störung eingesetzt. Im vorliegenden Fall soll dagegen die reale Ursache der Verdeckung nachgebildet werden.

Die Qualität eines Okkluders besitzt deshalb zwei Dimensionen:

1. **geometrische Wirkung:** Wie viel des Zielobjekts wird verdeckt?
2. **visuelle Plausibilität:** Wie ähnlich ist der Okkluder einem tatsächlich vor dem Objekt liegenden Profilelement?

Insbesondere die zweite Dimension kann relevant werden, wenn das Modell nicht nur das sichtbare Rohrende, sondern auch die charakteristischen visuellen Übergänge zwischen Objekt und Okkluder als Erkennungsmerkmal verwendet.

Ein künstlich erzeugter Okkluder kann dadurch selbst zu einer unerwünschten Trainingssignatur werden. Werden beispielsweise alle Verdeckungen mit identischen Kanten, Farben oder Schatten erzeugt, könnte das Modell lernen, diese künstlichen Muster mit der Zielklasse zu verbinden.

Die Hypothese dieser Arbeit lautet daher, dass nicht allein die Menge der Verdeckung, sondern auch deren visuelle Realistik die Übertragbarkeit auf reale Bilder beeinflusst. Diese Hypothese wird in der späteren Ablation experimentell überprüft.

---

## 3.5 Anforderungen an die Datengenerierungspipeline

Aus den vorangegangenen Überlegungen ergeben sich sechs Anforderungen an die in dieser Arbeit entwickelte Pipeline.

### A1 – Labelgenauigkeit

Geometrische Transformationen müssen mit einer möglichst geringen Abweichung zwischen Objekt und zugehöriger Bounding Box umgesetzt werden. Insbesondere dürfen Transformationen nicht unnötig zusätzliche Labelungenauigkeit erzeugen, wenn die zugrunde liegende Objektgeometrie im Generator bekannt ist.

### A2 – Kalibrierte Objektgrößen

Die erzeugte Größenverteilung muss die für die reale Inferenz relevante Objektgrößenverteilung hinreichend abdecken. Gleichzeitig dürfen durch Skalierung nicht systematisch Objekte erzeugt werden, deren effektive Größe im Verhältnis zur verwendeten Eingabeauflösung und zum feinsten Detektions-Stride zu klein ist.

### A3 – Physikalisch plausible Größenabhängigkeiten

Parameter, die in der realen Szene voneinander abhängig sind, dürfen nicht unabhängig randomisiert werden. Insbesondere müssen Objektgröße, perspektivische Darstellung und gegebenenfalls die Größe gemeinsam auftretender Instanzen konsistent behandelt werden.

### A4 – Kontrolle synthetischer Signaturen

Für die identifizierten Signaturen S1–S6 muss eine geeignete Gegenmaßnahme vorgesehen werden. Deren Wirksamkeit soll, soweit mit dem verfügbaren Validierungsdatensatz möglich, experimentell überprüft werden.

### A5 – Kontrollierbare Variation

Die Verteilung \(p(t)\) der Generatorparameter muss explizit dokumentiert und reproduzierbar sein. Die Parameter sollen dabei nicht nur technische Zufallswerte darstellen, sondern eine begründete Approximation der erwarteten Variation der Zieldomäne bilden.

### A6 – Trennung von Annahme und Messung

Eigenschaften der realen Domäne, die nicht direkt gemessen werden können, müssen als Annahmen gekennzeichnet werden. Parameter, deren Wahl ausschließlich auf einer theoretischen Überlegung oder Plausibilitätsannahme beruht, dürfen nicht als empirisch validierte Eigenschaften der Zieldomäne dargestellt werden.

Damit ergibt sich eine klare Trennung zwischen drei Ebenen der Untersuchung:

$$
\boxed{
\text{theoretische Anforderungen}
\rightarrow
\text{synthetische Datengenerierung}
\rightarrow
\text{experimentelle Validierung}
}
\tag{3.15}
$$

Die theoretische Analyse bestimmt somit nicht die optimale Augmentierung a priori. Sie definiert vielmehr die Eigenschaften, die eine geeignete Pipeline erfüllen sollte. Ob die daraus abgeleiteten Entscheidungen tatsächlich zu einer besseren Übertragung auf reale Bilder führen, wird in Kapitel 7 empirisch untersucht.
