#!/usr/bin/env python3
"""Pruefwerkzeug: Welche Zahl im Fliesstext hat keine Herkunftsangabe?

Zerlegt die Kapiteldateien in Saetze, sucht darin Zahlenwerte und prueft, ob
im selben Satz ein Herkunftsmarker steht -- also ein Verweis auf eine Messung,
eine Gleichung, eine Tabelle, eine Literaturstelle oder die ausdrueckliche
Kennzeichnung als gesetzter Wert bzw. Annahme.

Das Werkzeug ist eine *Erinnerungshilfe*, kein Urteil: Viele Treffer sind
korrekt, weil die Herkunft im Nachbarsatz oder in der Abschnittseinleitung
steht. Tabelle~\ref{tab:herkunft} im Anhang ist der verbindliche Nachweis.

Aufruf (aus paper/):   python3 tools/zahlen_audit.py kapitel
"""
import re, glob, os, sys
D = sys.argv[1] if len(sys.argv)>1 else 'kapitel'

def strip_tikz(s):
    out=[]; depth=0
    for line in s.split('\n'):
        if '\\begin{tikzpicture}' in line: depth+=1
        if depth>0:
            if '\\end{tikzpicture}' in line: depth-=1
            continue
        out.append(line)
    return '\n'.join(out)

MARK = re.compile(r'gemessen|Messung|Befund~|Tabelle~\\ref|\\eqref|parencite|textcite|'
                  r'gesetzt|Annahme|kalibriert|Erzeugungsbericht|Realmessung|'
                  r'Realbild|Realdaten|Abschnitt~\\ref|laut|Empfehlung|Standard', re.I)
ZAHL = re.compile(r'(?<![\w.,\\{])(\d+(?:[,.]\d+)?)(?![\w}])')
SKIP = re.compile(r'width=|scale=|includegraphics|\\label|itemsep|leftmargin|'
                  r'multicolumn|columntype|\\\\|toprule|midrule|bottomrule')
STRUKTUR = {'1','2','3','4','5','6','7','8','9'}   # Aufzaehlungen, Kopfzahlen

rows=[]
for f in sorted(glob.glob(f'{D}/0[1-8]*.tex')):
    s=strip_tikz(open(f).read())
    s=re.sub(r'\\z[A-Za-z]+','ZMACRO',s)
    for para in re.split(r'\n\s*\n', s):
        if SKIP.search(para) or para.strip().startswith('%'): continue
        flat=re.sub(r'\s+',' ',para)
        for satz in re.split(r'(?<=[.!?:;])\s+(?=[A-ZÄÖÜ\\])', flat):
            z=[x for x in ZAHL.findall(satz)]
            z=[x for x in z if x not in STRUKTUR or ',' in x or '.' in x]
            if not z: continue
            rows.append((os.path.basename(f), z, bool(MARK.search(satz)), satz[:160]))
mit=sum(1 for r in rows if r[2]); ohne=[r for r in rows if not r[2]]
print(f'{D}: {len(rows)} Saetze mit Zahl -- {mit} mit Herkunftsmarker, {len(ohne)} ohne\n')
for f,z,_,t in ohne:
    print(f'[{f:34s}] {z}')
    print(f'   {t}\n')
