"""
PipeNet - ein auf das konkrete Problem zugeschnittener Detektor.

ENTWURFSGRUNDLAGE (alle Zahlen gemessen, siehe ../docs/DATENSICHTUNG.md)
-----------------------------------------------------------------------
  * genau EINE Klasse, 9 zugrunde liegende Objektinstanzen
  * Objektdurchmesser 12-44 px, Median 20-27 px  -> enge Groessenspanne
  * Abstand zum naechsten Nachbarn 1.09-1.24 Durchmesser -> Objekte beruehren
    sich, ihre Zentren sind aber immer getrennt
  * Objekte sind Kreisringe, in Schraegansicht Ellipsen -> nahezu
    rotationssymmetrisch
  * Ziel ist Zaehlen, nicht pixelgenaue Segmentierung

ARCHITEKTURENTSCHEIDUNGEN, die daraus folgen
--------------------------------------------
1. EINE Ausgabeebene bei Stride 4 statt einer Merkmalspyramide.
   Rechnung bei 22 px Objektgroesse im Netzeingang:
       Stride  4 -> Objekt = 5.5 Zellen, Nachbarzentrum 5.5 Zellen entfernt
       Stride  8 -> Objekt = 2.8 Zellen, Nachbarzentrum 2.8 Zellen entfernt
       Stride 16 -> Objekt = 1.4 Zellen
       Stride 32 -> Objekt = 0.7 Zellen
   Stride 16 und 32 koennen ein 22-px-Objekt nicht mehr aufloesen; eine
   Pyramide ueber vier Ebenen verwaltet also drei Ebenen, von denen zwei
   fuer diese Aufgabe nutzlos sind. Bei einer Groessenspanne von nur
   12-44 px (Faktor 3.7) genuegt eine Ebene.

2. Encoder nach layer3 (Stride 16) abgeschnitten.
   ResNet18 hat 11.7 M Parameter, davon 8.4 M allein in layer4. Der
   rezeptive Bereich am Ende von layer3 ist bereits deutlich groesser als
   jedes Objekt samt Umgebung. Ergebnis: 2.78 M statt 11.2 M Encoder-
   Parameter. Das ist hier kein Sparzwang, sondern erwuenscht - bei nur 9
   Quellobjekten ist ueberschuessige Kapazitaet gleichbedeutend mit
   Auswendiglernen der synthetischen Texturen.

3. Zentrums-Heatmap statt Box-Regression mit NMS.
   Weil die Zentren immer getrennt sind, die Boxen sich aber ueberlappen,
   ist die Zentrumsdarstellung die passende: ein lokales Maximum = ein
   Objekt. Die Dekodierung braucht dann KEINE IoU-Schwelle (siehe
   decode.py), waehrend Box-NMS bei 0.84 Durchmessern Abstand im 10.
   Perzentil zwangslaeufig zwischen "Nachbar verlieren" und "Dublette
   behalten" abwaegen muss.

4. Kopf-Aufteilung: Heatmap (1) + log-Groesse (2) + Subpixel-Versatz (2).
   Der Versatzkopf holt die durch Stride 4 verlorene Genauigkeit zurueck
   (ohne ihn waere der Lokalisierungsfehler systematisch bis 2 px, was bei
   20-px-Objekten schon 0.2 IoU kostet). Die Groesse wird logarithmisch
   vorhergesagt, damit der L1-Verlust relative statt absoluter Fehler
   bestraft - bei einer Spanne von 12-44 px ist das der richtige Bezug.

5. Ein zusaetzlicher Skip von der Stride-2-Ebene des Stems.
   Das diskriminative Merkmal ist der 1-3 px breite helle Rohrkranz. Die
   Stride-2-Merkmale des Stems haben ihn noch, layer1 (nach MaxPool) nur
   noch abgeschwaecht. Der Skip kostet 37 k Parameter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

OUT_STRIDE = 4        # Ausgabe-Stride, siehe Entwurfsgrundlage Punkt 1
HEAD_CH = 64


def _conv_bn_relu(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Head(nn.Module):
    """3x3 Faltung + 1x1 Projektion, wie in CenterNet."""

    def __init__(self, cin, cout, bias_init=None):
        super().__init__()
        self.block = _conv_bn_relu(cin, HEAD_CH)
        self.out = nn.Conv2d(HEAD_CH, cout, 1)
        if bias_init is not None:
            nn.init.constant_(self.out.bias, bias_init)
            nn.init.normal_(self.out.weight, std=0.001)

    def forward(self, x):
        return self.out(self.block(x))


class PipeNet(nn.Module):
    """Encoder (ResNet18 bis layer3) + Top-Down-Dekoder auf Stride 4 + 3 Koepfe.

    Ausgabe (jeweils H/4 x W/4):
        hm   1 Kanal, Logits  -> Zentrums-Heatmap
        wh   2 Kanaele        -> log(Breite in px), log(Hoehe in px)
        off  2 Kanaele        -> Subpixel-Versatz in Zelleinheiten [0,1)
    """

    def __init__(self, pretrained=True, width=64):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = resnet18(weights=w)
        # --- Encoder-Abgriffe -------------------------------------------
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)   # Stride 2,  64
        self.pool = r.maxpool                                # Stride 4
        self.layer1 = r.layer1                               # Stride 4,  64
        self.layer2 = r.layer2                               # Stride 8, 128
        self.layer3 = r.layer3                               # Stride 16,256
        # layer4 (Stride 32, 8.4 M Parameter) wird bewusst NICHT verwendet

        # --- Lateralprojektionen auf gemeinsame Breite -------------------
        self.lat3 = nn.Conv2d(256, width, 1)
        self.lat2 = nn.Conv2d(128, width, 1)
        self.lat1 = nn.Conv2d(64, width, 1)
        self.smooth2 = _conv_bn_relu(width, width)
        self.smooth1 = _conv_bn_relu(width, width)
        # Skip von Stride 2 auf Stride 4 herunterfalten (Entwurfspunkt 5)
        self.fine = _conv_bn_relu(64, width, k=3, s=2)
        self.fuse = _conv_bn_relu(width, width)

        # --- Koepfe ------------------------------------------------------
        # Bias -4.595 = logit(0.01): zu Beginn sagt das Netz ueberall
        # "kein Objekt" voraus. Ohne diesen Prior dominieren in den ersten
        # Iterationen die ~65000 negativen Zellen je Bild den Verlust.
        self.hm = Head(width, 1, bias_init=-4.595)
        self.wh = Head(width, 2)
        self.off = Head(width, 2)

    def forward(self, x):
        s2 = self.stem(x)                  # Stride 2
        c1 = self.layer1(self.pool(s2))    # Stride 4
        c2 = self.layer2(c1)               # Stride 8
        c3 = self.layer3(c2)               # Stride 16

        p3 = self.lat3(c3)
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode="nearest"))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(
            p2, size=c1.shape[-2:], mode="nearest"))
        f = self.fuse(p1 + self.fine(s2))  # Stride 4, width Kanaele

        return {"hm": self.hm(f), "wh": self.wh(f), "off": self.off(f)}

    # -- Hilfsfunktionen ---------------------------------------------------
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, wd=1e-4):
        """Gewichtsdaempfung nur auf Faltungs-/Linear-Gewichte, nicht auf
        BatchNorm-Parameter und Biases - Standardpraxis, verhindert dass die
        Normierungsstatistik gegen Null gezogen wird."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        return [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]


if __name__ == "__main__":
    m = PipeNet(pretrained=False)
    x = torch.zeros(1, 3, 1024, 1024)
    with torch.no_grad():
        o = m(x)
    print(f"Parameter: {m.n_params() / 1e6:.2f} M")
    for k, v in o.items():
        print(f"  {k:4s} {tuple(v.shape)}  (Stride "
              f"{x.shape[-1] // v.shape[-1]})")
