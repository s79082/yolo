# PipeNet ganz einfach erklärt

> **Hinweis:** Dies ist die erste, stark vereinfachte Fassung (Zielgruppe ~10
> Jahre). Die Fassung fuer Informatikstudierende steht in `../einfach.md`.

*Für alle, die keine Lust auf Fachwörter haben — zum Beispiel für einen
10-Jährigen.*

![Wie PipeNet Rohre findet](fig_einfach.png)

---

## Was soll der Computer eigentlich machen?

Stell dir einen großen Haufen Bauzäune vor. So welche, wie sie um eine
Baustelle stehen. Die liegen aufeinander gestapelt, hunderte davon.

Jeder Bauzaun hat einen Rahmen aus Rohren. Wenn du von der Seite auf den
Stapel schaust, siehst du die **Enden** dieser Rohre — kleine dunkle Ringe,
wie wenn du in ein Bündel Strohhalme von vorne hineinschaust.

**Die Aufgabe:** Der Computer soll auf einem Foto alle diese Rohrenden finden
und zählen.

Warum? Weil jemand wissen will, wie viele Bauzäune im Lager liegen. Von Hand
zählen dauert lang und man verzählt sich.

---

## Warum ist das schwer?

Drei Dinge machen es schwer:

**1. Die Rohrenden sind winzig.** Auf dem Foto sind sie nur etwa 20 Pixel
groß. Ein Pixel ist ein einzelner Bildpunkt. 20 Pixel — das ist etwa so
groß wie der Punkt über dem i, wenn du dieses Blatt ausdruckst.

**2. Sie liegen dicht an dicht.** Sie berühren sich sogar. Wie Strohhalme in
einem Bündel.

**3. Manche sind halb versteckt.** Vor dem Stapel liegt oft noch ein Rohr
quer und verdeckt die Hälfte.

---

## Wie bringt man einem Computer sowas bei?

Nicht mit Regeln. Man kann nicht sagen „such einen dunklen Kreis" — dann
findet er auch Löcher im Gitter, Schatten und Steine.

Stattdessen zeigt man ihm **ganz viele Beispiele**. Man sagt: „Hier ist ein
Foto, und hier, hier und hier sind Rohrenden drauf." Das macht man tausende
Male. Irgendwann merkt sich der Computer, wie ein Rohrende aussieht. Das
nennt man **trainieren**.

### Aber wir hatten fast keine Beispiele!

Und jetzt kommt das Problem: Wir hatten nur **9** ausgeschnittene Rohrenden.
Neun! Wie neun Sticker. Dazu ein paar Fotos von Rohrstapeln, auf denen man
gerade *keine* Enden sieht.

Also haben wir **gebastelt**: Wir haben die 9 Sticker tausende Male auf die
Fotos geklebt. Jedes Mal ein bisschen anders:

* mal gedreht,
* mal heller, mal dunkler,
* mal größer, mal kleiner,
* mal mit einem Rohr davor, das die Hälfte verdeckt,
* mal ein bisschen eingedellt, als wäre es kaputt.

So sind **6000 Übungsbilder** entstanden. Alle selbst gebastelt.

### Der wichtigste Trick beim Basteln

Man darf **nicht sehen**, dass die Sticker aufgeklebt sind. Keine Klebekante,
kein Helligkeitssprung, nichts.

Warum? Weil der Computer sonst schummelt. Er würde lernen: „Such die
Klebekante!" statt „Such ein Rohrende!" Auf den Bastelbildern wäre er dann
super — auf echten Fotos aber völlig hilflos.

Das ist wie bei einem Test, wo du merkst, dass die richtige Antwort immer die
längste ist. Du bekommst eine Eins, hast aber nichts verstanden.

---

## Und wie funktioniert PipeNet jetzt?

### Schritt 1: Erst wegzoomen, dann wieder ranzoomen

PipeNet schaut sich das Bild **viermal** an. Jedes Mal etwas gröber — so als
würdest du die Augen immer mehr zusammenkneifen.

* Beim ersten Mal sieht es nur **Striche und Kanten**.
* Dann setzen sich die Striche zu **Ringen** zusammen.
* Dann erkennt es ganze **Bereiche**: „hier ist ein Stapel, hier ist Erde".

Das ist wie bei Lego: erst einzelne Steine, dann eine Mauer, dann ein Haus.

Danach macht PipeNet das Bild wieder **feiner**. Denn jetzt muss es ja genau
zeigen können, *wo* die Rohrenden sind. Ungefähr so, wie wenn du erst aus dem
Flugzeug auf eine Stadt schaust, um dich zu orientieren, und dann wieder
runterzoomst, um auf ein einzelnes Haus zu zeigen.

### Schritt 2: Die Schatzkarte

Jetzt kommt die schönste Idee.

PipeNet malt eine **ganz schwarze Karte**, genauso groß wie das Foto. Und
überall, wo es ein Rohrende sieht, macht es einen **leuchtenden Punkt** hin.
Wie ein X auf einer Schatzkarte.

Am Ende muss man nur noch die leuchtenden Punkte zählen. Fertig.

### Warum Punkte und keine Kästchen?

Die meisten Programme malen um jedes Ding einen **Kasten**. Das klappt gut bei
Autos oder Hunden — die stehen ja nicht ineinander.

Aber unsere Rohre berühren sich. Wenn man um jedes einen Kasten malt,
**überlappen sich die Kästen**, und dann weiß niemand mehr, welcher Kasten zu
welchem Rohr gehört. Das Programm muss dann raten und wirft manchmal ein
echtes Rohr weg.

Die **Mittelpunkte** dagegen liegen niemals aufeinander. Nie. Auch wenn die
Rohre sich berühren, haben ihre Mitten immer ein Stück Abstand.

Darum sucht PipeNet **Mittelpunkte statt Kästen**. Das ist der ganze Trick.

### Schritt 3: Das Karopapier muss fein sein

Die Schatzkarte ist eigentlich **Karopapier**. Jedes Kästchen deckt 4 mal 4
Pixel vom Foto ab.

Das ist wichtig: Wären die Kästchen größer — sagen wir 8 mal 8 — dann würden
zwei benachbarte Rohre ins **gleiche Kästchen** fallen. Und aus zwei Rohren
würde ein einziger Punkt. Der Computer hätte sich verzählt.

Mit den kleinen Kästchen liegen zwischen zwei Rohrmitten immer etwa **5
Kästchen** Platz. Das reicht, um sie auseinanderzuhalten.

### Schritt 4: Der Feinschliff

Karopapier ist praktisch, aber ungenau. Ein Rohrende sitzt ja selten genau in
einer Kästchen-Ecke.

Deshalb sagt PipeNet zusätzlich: *„Der Mittelpunkt liegt nicht genau hier,
sondern ein kleines Stück weiter rechts und ein Stück weiter unten."* Damit
kann es **genauer zeigen, als das Karopapier eigentlich erlaubt**.

Und es sagt noch: *„Das Rohr ist etwa so breit und so hoch."* Damit kann man
am Ende doch noch einen Kasten drum malen, wenn man einen braucht.

---

## Zwei Sachen, die schlau sind

### Absichtlich ein kleines Gehirn

PipeNet hat rund **3 Millionen kleine Stellschrauben**. Das klingt nach
irrsinnig viel, ist aber **dreimal weniger** als bei den fertigen Programmen,
gegen die wir es verglichen haben.

Klein ist hier **absichtlich**. Wir hatten ja nur 9 echte Rohrenden. Ein
riesiges Gehirn hätte einfach diese 9 Bilder **auswendig gelernt**, statt zu
verstehen, wie ein Rohrende überhaupt aussieht. Wie wenn du die Lösungen
abschreibst statt zu rechnen: klappt bei der Hausaufgabe, nicht in der
Klassenarbeit.

### Viermal hinschauen

Ein Rohrende sieht von allen Seiten gleich aus. Dreh das Foto um 90 Grad —
es ist immer noch ein Ring.

Darum darf PipeNet zum Schluss **achtmal** hinschauen: normal, um 90 Grad
gedreht, um 180, um 270, und alles noch mal gespiegelt. Dann nimmt es den
Mittelwert. So wird es sicherer.

---

## Der peinlichste Fehler — und wie wir ihn gefunden haben

Als wir uns die Schatzkarten angeschaut haben, saßen die leuchtenden Punkte
manchmal **nicht ganz** auf den Rohrenden, sondern ein Stück **darüber** oder
**daneben**.

Erster Gedanke: Mist, PipeNet zeigt falsch hin.

Aber dann haben wir nachgemessen — und PipeNet hatte **recht**. Die Punkte
stimmten auf weniger als einen Pixel genau. Wir hatten sie nur **falsch
aufgemalt**! Beim Übereinanderlegen von Schatzkarte und Foto hatten wir uns
verrechnet, und zwar um bis zu 7 Pixel. Bei einem 20 Pixel kleinen Rohr ist
das über ein Drittel daneben.

Die Lehre daraus: Wenn etwas komisch aussieht, muss nicht das Programm falsch
sein. Manchmal ist nur die **Brille** falsch, durch die man schaut. Und
herausfinden kann man das nur, wenn man **nachmisst** statt zu raten.

---

## Und, funktioniert es?

Auf einem Foto, das PipeNet **noch nie gesehen hatte**, sind 86 Rohrenden.
PipeNet hat 82 davon richtig gefunden.

Das sind ungefähr **9 von 10**. Und das, obwohl es nur mit selbst gebastelten
Bildern geübt hat und nie ein einziges echtes Übungsfoto gesehen hat.

---

## Kleines Wörterbuch

| Wort | Was es bedeutet |
|---|---|
| **Pixel** | Ein einzelner Bildpunkt. Ein Foto besteht aus Millionen davon. |
| **Neuronales Netz** | Ein Programm, das aus Beispielen lernt, statt festen Regeln zu folgen. |
| **trainieren** | Dem Netz ganz viele Beispiele zeigen, bis es die Sache kann. |
| **Heatmap** | Unsere „Schatzkarte": ein schwarzes Bild mit leuchtenden Punkten. |
| **Stride 4** | Die Größe der Karo-Kästchen: eins deckt 4 × 4 Pixel ab. |
| **Parameter** | Die „Stellschrauben" im Netz. PipeNet hat 3 Millionen davon. |
| **Sticker** | Hier gemeint: ein ausgeschnittenes echtes Rohrende zum Aufkleben. |

---

*Wenn du wissen willst, wie das Ganze technisch genau aussieht:
[`ARCHITEKTUR.md`](ARCHITEKTUR.md) und die große Grafik
[`fig_architektur.png`](fig_architektur.png).*
