# Conference talk

Beamer deck for "Mind the Gap: Joint vs. Data-Incremental Self-Supervised Pretraining
for Plant Classification", prepared for the GIL-Jahrestagung 2027.

## Build

```bash
export PATH=/cluster/apps/texlive/2026/bin/x86_64-linux:$PATH
pdflatex slides && biber slides && pdflatex slides && pdflatex slides
```

Use **pdflatex**, not lualatex or xelatex. Metropolis prefers `fontspec`, but Fira Sans and
Roboto are not installed as system fonts here; the `roboto` package supplies the house
typeface for pdflatex instead.

The bibliography is read from `../paper/bibliography.bib`, so the talk and the paper cite
the same verified entries. Citation style is `authoryear`, which reads better from a
projector than the paper's alphabetic LNI labels.

## Corporate design

Colours and typeface come from `HSOS_CD_Manual_2026-08.pdf`:

| Element | Source | Value |
|---|---|---|
| Accent | p. 17, green tone 06, main colour "Forschung" | `#228B81` |
| Logo cyan | p. 15, Pantone Process Cyan C | `#009EE3` |
| Text | p. 15, neutral grey | `#333333` |
| Greys | p. 15, Pantone Cool Gray 11C / 6C | `#5E5E5D`, `#CBCCCC` |
| Typeface | p. 22, Hausschrift | Roboto |

The manual states that HEX and RGB are the online values while CMYK are the print values,
so a screen deck uses the HEX values above verbatim. The research-green accent is used
because marker 06 in the manual is the main colour for "Forschung".

## Structure

19 frames: title, motivation and the joint-vs-incremental decision, the research question,
dataset and compute-matched design, evaluation protocol, four results frames, mechanism,
takeaways, limitations and future work, references, and a closing slide.

Every number shown is checked against `../results/*.csv`; the two charts carry the same
coordinates as the paper's figures.
