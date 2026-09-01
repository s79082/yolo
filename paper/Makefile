MAIN  := main
KURZ  := main_p2_kurz
E2    := main_e2
BUILD := build

# latexmk uebernimmt die Abhaengigkeitsverfolgung selbst (ueber .fls/.fdb) und
# entscheidet, wie oft uebersetzt und wann biber aufgerufen werden muss.  Die
# Ziele sind deshalb bewusst .PHONY und tragen KEINE Abhaengigkeitsliste --
# eine handgeschriebene Liste waere nur eine schlechtere Kopie davon und fuehrt
# zu "Nothing to be done", sobald ein PDF zufaellig neuer ist als die Quellen.
LATEXMK := latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=$(BUILD)

.PHONY: all kurz e2 beide manual abbildungen pruefen clean distclean

all:
	@mkdir -p $(BUILD)
	$(LATEXMK) $(MAIN).tex
	@cp $(BUILD)/$(MAIN).pdf ./$(MAIN).pdf
	@echo "--> $(MAIN).pdf"

kurz:
	@mkdir -p $(BUILD)
	$(LATEXMK) $(KURZ).tex
	@cp $(BUILD)/$(KURZ).pdf ./$(KURZ).pdf
	@echo "--> $(KURZ).pdf"

e2:
	@mkdir -p $(BUILD)
	$(LATEXMK) $(E2).tex
	@cp $(BUILD)/$(E2).pdf ./$(E2).pdf
	@echo "--> $(E2).pdf"

beide: all kurz

# Fallback ohne latexmk: vier Durchlaeufe im aktuellen Verzeichnis.
manual:
	pdflatex -interaction=nonstopmode $(MAIN).tex
	biber $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Zeigt die Warnungen, die man vor der Abgabe gesehen haben will.
pruefen:
	@for f in $(MAIN) $(KURZ) $(E2); do \
	  test -f $(BUILD)/$$f.log || continue; \
	  echo "=== $$f"; \
	  grep -c "undefined" $(BUILD)/$$f.log | sed 's/^/  undefined: /'; \
	  grep -c "Overfull" $(BUILD)/$$f.log | sed 's/^/  Overfull:  /'; \
	done

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
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).bcf $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).run.xml \
	      $(MAIN)-blx.bib $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz
	rm -f $(KURZ).aux $(KURZ).bbl $(KURZ).bcf $(KURZ).blg $(KURZ).log \
	      $(KURZ).out $(KURZ).toc $(KURZ).lof $(KURZ).lot $(KURZ).run.xml \
	      $(KURZ)-blx.bib $(KURZ).fls $(KURZ).fdb_latexmk $(KURZ).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf $(KURZ).pdf
