"""Laeufe und Datensaetze, auf die sich Auswertung und Bericht beziehen.

Die Arbeit berichtet ein Referenzmodell und die beiden Arme der Ablation C.
Jeder Eintrag nennt den Laufordner unter runs/ und den Trainingsdatensatz,
die Arme zusaetzlich ihren eigenen synthetischen Validierungssatz: Auf ihm
zeigt sich, ob ein Arm seine Trainingsdaten gelernt hat.

Gelesen wird diese Datei von src/evaluate.py (Split sval),
src/experiment_neu.py (main, sweep), src/analyse_neu.py, src/eval_armC.py
und src/report_neu.py.
"""
import os

REFERENZ = dict(lauf="bereinigt_referenz", daten="synth/bereinigt_train")

# sval des Referenzmodells; zugleich der eigene sval von C0, weil beide mit
# der vollstaendigen Pipeline und demselben Seed erzeugt sind.
SVAL = "synth/bereinigt_sval"

ARME_C = {
    "C0_kontrolle": dict(
        lauf="bereinigt_C0_kontrolle", daten="synth/bereinigt_C0_kontrolle",
        sval=SVAL,
        beschreibung="Kontrolle: vollstaendige Pipeline, mit demselben Code "
                     "erzeugt wie C1"),
    "C1_naive_paste": dict(
        lauf="bereinigt_C1_naive_paste", daten="synth/bereinigt_C1_naive_paste",
        sval="synth/bereinigt_sval_C1",
        beschreibung="naives Einkleben: Gegenmassnahmen zu den Signaturen "
                     "S1-S5 abgeschaltet"),
}


def gewichte(lauf):
    """Bester Checkpoint eines Laufs, relativ zum Projektwurzelverzeichnis."""
    return os.path.join("runs", lauf, "weights", "best.pt")
