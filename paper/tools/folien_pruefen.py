#!/usr/bin/env python3
"""Meldet Folien, deren Inhalt nicht auf die Folie passt.

Der Log von pdflatex nennt bei einem Ueberlauf nur die Zeilennummer in
vortrag.tex.  Dieses Skript uebersetzt sie in den Folientitel -- sonst sucht
man bei vierzig Folien die betroffene von Hand.

    python3 tools/folien_pruefen.py [build/vortrag.log] [vortrag.tex]

Rueckgabewert 1, sobald eine Folie ueber der Toleranz liegt; damit laesst es
sich in ein Makefile-Ziel haengen.
"""
import re
import sys

# Unter dieser Grenze ist ein Ueberlauf ein Bruchteil einer Zeile und auf dem
# Projektor nicht zu sehen.  Alles darueber schneidet sichtbar ab.
TOLERANZ_PT = 4.0

VBOX = re.compile(r"Overfull \\vbox \(([0-9.]+)pt too high\) detected at line (\d+)")
HBOX = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\) in paragraph at lines (\d+)")


def folien_index(quelle):
    """Zeilennummer -> Titel der umgebenden Folie.

    Bewusst OHNE laufende Nummer: \\trenner erzeugt zusaetzliche Rahmen und
    [noframenumbering] zaehlt nicht mit, eine hier gezaehlte Nummer stimmte
    also weder mit der PDF-Seite noch mit der Fussnote der Folie ueberein.
    Der Titel ist eindeutig und im Editor suchbar -- das genuegt.
    """
    index = []
    titel = "(vor der ersten Folie)"
    start = 1
    with open(quelle, encoding="utf-8") as fh:
        for n, zeile in enumerate(fh, 1):
            k = zeile.lstrip()
            if k.startswith("\\begin{frame}") or k.startswith("\\trenner"):
                m = re.search(r"\\begin\{frame\}(?:\[[^\]]*\])?\{(.*?)\}", zeile)
                if m:
                    neu = m.group(1)
                elif k.startswith("\\trenner"):
                    t = re.findall(r"\{([^{}]*)\}", zeile)
                    neu = "Trenner: " + " / ".join(t) if t else "Trenner"
                else:
                    neu = "(ohne Titel)"
                index.append((start, n - 1, titel))
                titel, start = neu, n
    index.append((start, 10**9, titel))
    return index


def zuordnen(index, zeile):
    for von, bis, titel in index:
        if von <= zeile <= bis:
            return titel
    return "(unbekannt)"


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "build/vortrag.log"
    quelle = sys.argv[2] if len(sys.argv) > 2 else "vortrag.tex"

    try:
        text = open(log, encoding="utf-8", errors="replace").read()
    except OSError:
        print(f"kein Log unter {log} -- erst 'make vortrag' aufrufen")
        return 1

    index = folien_index(quelle)
    treffer = []
    for muster, art in ((VBOX, "zu hoch"), (HBOX, "zu breit")):
        for m in muster.finditer(text):
            pt, zeile = float(m.group(1)), int(m.group(2))
            treffer.append((pt, art, zuordnen(index, zeile), zeile))

    fehler = [t for t in treffer if t[0] >= TOLERANZ_PT]
    dulden = [t for t in treffer if t[0] < TOLERANZ_PT]

    if not treffer:
        print("  alle Folien passen")
    for pt, art, titel, zeile in sorted(fehler, key=lambda t: -t[0]):
        print(f"  {pt:6.1f} pt {art}   {titel}   (vortrag.tex:{zeile})")
    if dulden:
        print(f"  {len(dulden)} Ueberlauf unter {TOLERANZ_PT:.0f} pt -- unsichtbar, geduldet")

    if fehler:
        print(f"\n  {len(fehler)} Folie(n) ueber der Toleranz von {TOLERANZ_PT:.0f} pt.")
        print("  Kuerzen, nicht verkleinern: der Sprechtext in \\note{} traegt die Prosa.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
