MAIN  := main
BUILD := build

# kapitel/ enthaelt alles: Fliesstext, zahlen.tex und die tab_*.tex, die
# src/report_neu.py und src/analyse_ablationB.py dorthin schreiben.

# latexmk uebernimmt die Abhaengigkeitsverfolgung selbst (ueber .fls/.fdb) und
# entscheidet, wie oft uebersetzt und wann biber aufgerufen werden muss.  Die
# Ziele sind deshalb bewusst .PHONY und tragen KEINE Abhaengigkeitsliste --
# eine handgeschriebene Liste waere nur eine schlechtere Kopie davon und fuehrt
# zu "Nothing to be done", sobald ein PDF zufaellig neuer ist als die Quellen.
LATEXMK := latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=$(BUILD)

.PHONY: all vortrag vortrag-notizen vortrag-pruefen manual abbildungen pruefen clean distclean

all:
	@mkdir -p $(BUILD)
	$(LATEXMK) $(MAIN).tex
	@cp $(BUILD)/$(MAIN).pdf ./$(MAIN).pdf
	@echo "--> $(MAIN).pdf"

# ---------------------------------------------------------------------------
#  Verteidigungsvortrag.  vortrag.tex liest dieselbe kapitel/zahlen.tex wie
#  main.tex -- die Folien koennen deshalb nicht von der Arbeit abweichen.
#  Die Folien tragen Stichpunkte, die \note{}-Bloecke den Sprechtext;
#  der Ablaufplan steht in VORTRAG.md.
# ---------------------------------------------------------------------------
vortrag:
	@mkdir -p $(BUILD)
	$(LATEXMK) vortrag.tex
	@cp $(BUILD)/vortrag.pdf ./vortrag.pdf
	@echo "--> vortrag.pdf"

# Nennt die Folien, deren Inhalt nicht auf die Folie passt -- der Log von
# pdflatex gibt dafuer nur Zeilennummern her.
vortrag-pruefen: vortrag
	@python3 tools/folien_pruefen.py $(BUILD)/vortrag.log vortrag.tex

# Fassung mit Sprechnotizen: doppelt breite Seiten, Folie links, Notiz rechts.
# Fuer den Praesentationsmodus (zweiter Bildschirm) oder zum Ausdrucken.
# \NOTIZEN wird von aussen gesetzt, damit nur eine Quelldatei zu pflegen ist;
# eigenes -jobname, damit .aux und .fdb der beiden Fassungen sich nicht in die
# Quere kommen.
vortrag-notizen:
	@mkdir -p $(BUILD)/notizen
	pdflatex -interaction=nonstopmode -file-line-error \
	         -output-directory=$(BUILD)/notizen -jobname=vortrag_notizen \
	         '\def\NOTIZEN{}\input{vortrag}'
	pdflatex -interaction=nonstopmode -file-line-error \
	         -output-directory=$(BUILD)/notizen -jobname=vortrag_notizen \
	         '\def\NOTIZEN{}\input{vortrag}'
	@cp $(BUILD)/notizen/vortrag_notizen.pdf ./vortrag_notizen.pdf
	@echo "--> vortrag_notizen.pdf"

# Fallback ohne latexmk: vier Durchlaeufe im aktuellen Verzeichnis.
manual:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	biber $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Zeigt die Warnungen, die man vor der Abgabe gesehen haben will.
pruefen:
	@test -f $(BUILD)/$(MAIN).log || { echo "erst 'make' aufrufen"; exit 1; }
	@grep -c "undefined" $(BUILD)/$(MAIN).log | sed 's/^/  undefined: /'
	@grep -c "Overfull"  $(BUILD)/$(MAIN).log | sed 's/^/  Overfull:  /'

# Abbildungen aus den Rohdaten des Repositorys neu aufbereiten.
# Aus dem Projektwurzelverzeichnis aufzurufen, weil die Skripte relative
# Pfade auf docs/ und den Modellgewichten verwenden.
abbildungen:
	cd .. && .venv/bin/python paper/tools/abbildungen_vorbereiten.py
	cd .. && .venv/bin/python paper/tools/fig_p2.py
	cd .. && .venv/bin/python paper/tools/fig_boxversatz.py
	cd .. && .venv/bin/python paper/tools/fig_qual_iou.py
	cd .. && .venv/bin/python paper/tools/fig_stufen.py

clean:
	rm -rf $(BUILD)
	rm -f vortrag.aux vortrag.log vortrag.nav vortrag.out vortrag.snm \
	      vortrag.toc vortrag.fls vortrag.fdb_latexmk
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).bcf $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).run.xml \
	      $(MAIN)-blx.bib $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf vortrag.pdf vortrag_notizen.pdf
