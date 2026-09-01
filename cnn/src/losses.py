"""
Verlustfunktionen fuer PipeNet.

Drei Terme, jeder mit einer konkreten Begruendung fuer diese Aufgabe.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_focal_loss(logits, target, alpha=2.0, beta=4.0, eps=1e-6):
    """Punktweiser Focal Loss auf einer gauss-verschmierten Zielheatmap.

    Variante aus CornerNet/CenterNet. Warum nicht einfach BCE:
    In einem 256x256-Ausgabegitter mit ~23 Objekten sind 65 513 von 65 536
    Zellen negativ (99.96 %). Reines BCE wird vollstaendig von den Negativen
    dominiert. Die beiden Exponenten wirken dagegen:

      alpha = 2  daempft leichte Beispiele (bereits richtig vorhergesagte
                 Zellen tragen kaum noch bei)
      beta  = 4  daempft Negative in der NAEHE eines Zentrums, wo das Ziel
                 wegen der Gauss-Verschmierung nahe 1 liegt. Ohne diesen Term
                 wuerde das Netz dafuer bestraft, einen Nachbarpixel des
                 Zentrums ebenfalls hoch zu bewerten - genau die Toleranz,
                 die man bei 5.5 Zellen grossen Objekten braucht.

    Normiert wird durch die Anzahl echter Zentren, nicht durch die Zellzahl:
    sonst haengt die Verlustgroesse von der Bildgroesse ab.
    """
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def masked_l1(pred, target, mask):
    """L1 nur an den Zellen echter Zentren.

    Warum maskiert: an allen anderen Zellen ist die Groesse undefiniert. Ein
    unmaskierter Verlust wuerde das Netz zwingen, ueberall irgendeine Groesse
    vorherzusagen, und der Gradient der 99.96 % undefinierten Zellen wuerde
    das Signal der 0.04 % definierten erschlagen.
    """
    n = mask.sum()
    if n == 0:
        return pred.sum() * 0.0
    m = mask.bool().unsqueeze(1).expand_as(pred)
    return F.l1_loss(pred[m], target[m], reduction="sum") / n


class PipeLoss(torch.nn.Module):
    """L = L_hm + w_wh * L_wh + w_off * L_off

    Gewichte:
      w_wh  = 1.0  Die Groesse wird logarithmisch vorhergesagt. Der typische
                   L1-Fehler liegt damit im Bereich 0.05-0.3 und ist von sich
                   aus vergleichbar zum Heatmap-Term; die in CenterNet
                   ueblichen 0.1 gelten fuer Groessen in Pixeleinheiten
                   (Fehler ~2-5) und waeren hier zwei Groessenordnungen zu
                   klein.
      w_off = 1.0  Der Versatz liegt in [0,1); Fehler ~0.1-0.3, also
                   dieselbe Groessenordnung.
    """

    def __init__(self, w_wh=1.0, w_off=1.0, alpha=2.0, beta=4.0):
        super().__init__()
        self.w_wh, self.w_off = w_wh, w_off
        self.alpha, self.beta = alpha, beta

    def forward(self, pred, tgt):
        l_hm = gaussian_focal_loss(pred["hm"], tgt["hm"],
                                   self.alpha, self.beta)
        l_wh = masked_l1(pred["wh"], tgt["wh"], tgt["mask"])
        l_off = masked_l1(pred["off"], tgt["off"], tgt["mask"])
        total = l_hm + self.w_wh * l_wh + self.w_off * l_off
        with torch.no_grad():
            parts = {"hm": l_hm.item(), "wh": l_wh.item(),
                     "off": l_off.item(), "total": total.item()}
        return total, parts
