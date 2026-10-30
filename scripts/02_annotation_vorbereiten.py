#!/usr/bin/env python3
"""Schritt 2: Label-Studio-Projekt fuer die Punktannotation vorbereiten.

Annotiert wird der BLUETENCLUSTER (Bluetenbueschel), nicht die Einzelbluete.
Begruendung: Das ist die agronomisch relevante Einheit fuer die Ausduennung und
die Zaehleinheit, die auch Farjon et al. (2020) und Shang et al. (2024) verwenden.
Punkte statt Boxen, weil Punktannotation bei dichten Kleinobjekten rund 5-10x
schneller geht; die Umrechnung in Pseudo-Boxen fester Groesse macht Schritt 3.

Label Studio laeuft lokal - die Bilder verlassen den Rechner nicht. Das ist bei
einem Paper zur digitalen Souveraenitaet kein Detail, sondern Teil der Aussage.

    python3 02_annotation_vorbereiten.py
    bash ../annotation/label_studio_starten.sh
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent.parent

LABEL_CONFIG = """<View>
  <!-- Zoom ist zwingend: die Cluster sind bei 1024px-Kacheln klein -->
  <Image name="bild" value="$image" zoom="true" zoomControl="true"
         rotateControl="false" width="100%"/>
  <KeyPointLabels name="punkte" toName="bild" strokeWidth="2" opacity="0.9">
    <Label value="Bluetencluster" background="#e6194b"/>
    <Label value="unsicher" background="#ffe119"/>
  </KeyPointLabels>
</View>
"""

STARTSKRIPT = """#!/usr/bin/env bash
# Startet Label Studio lokal mit Zugriff auf den Kachelordner.
set -euo pipefail

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="{kachelwurzel}"

echo "Label Studio startet auf http://localhost:8080"
echo "Dokumentwurzel: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
label-studio start --data-dir "{datadir}"
"""

ANLEITUNG = """
So legst du das Projekt in Label Studio an
==========================================

1.  Browser auf http://localhost:8080, Konto anlegen (rein lokal, keine Cloud).

2.  "Create Project" -> Name z.B. "Bluete Ablass 5m".

3.  Reiter "Labeling Setup" -> "Custom template" -> Inhalt von
        {config}
    einfuegen und speichern.

4.  Reiter "Data Import" -> "Upload Files" waehlen und die Kacheln aus
        {kachelordner}
    hochladen. (Bei bis zu ein paar hundert Kacheln ist Upload einfacher als
    der Local-Storage-Umweg; die Dateien bleiben trotzdem auf dem Rechner.)

5.  Annotieren. Tastaturkuerzel: Ziffer 1 waehlt "Bluetencluster",
    Klick setzt einen Punkt, Strg+Z macht rueckgaengig.

    Annotationsregeln - VORHER festlegen und im Paper berichten:
      * Ein Punkt pro Bluetencluster, gesetzt in die Clustermitte.
      * Angeschnittene Cluster am Kachelrand: zaehlen, wenn ueber die Haelfte
        sichtbar ist.
      * Cluster im Schatten: zaehlen, wenn als Cluster erkennbar.
      * Zweifelsfall -> Label "unsicher". Diese fliegen im Training raus,
        aber ihre ANZAHL ist ein Ergebnis: sie misst, wie eindeutig die
        Aufgabe bei dieser Aufloesung ueberhaupt ist.

6.  WICHTIG fuer die Reproduzierbarkeit: 20 Kacheln nach einer Woche ein
    zweites Mal annotieren (neues Projekt, gleiche Regeln, alte Annotation
    nicht ansehen). Daraus die Intra-Rater-Uebereinstimmung berechnen.
    Das Gutachtenkriterium "Verifizierbarkeit und Transparenz" fragt genau
    danach - und dein eigener Zauberwald-Vermerk nennt die Subjektivitaet der
    Bluehstaerkebewertung als Kernproblem. Sie zu messen statt zu beklagen ist
    ein Ergebnis.

7.  Export: "Export" -> Format "JSON" -> speichern unter
        {export}

Dann weiter mit:
    python3 03_training.py
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kachelordner", type=Path, default=None)
    args = p.parse_args()

    annot = PROJEKT / "annotation"
    annot.mkdir(parents=True, exist_ok=True)

    kachelwurzel = PROJEKT / "kacheln"
    kachelordner = args.kachelordner
    if kachelordner is None:
        unterordner = [d for d in sorted(kachelwurzel.glob("*")) if d.is_dir()]
        kachelordner = unterordner[0] if unterordner else kachelwurzel

    config = annot / "labelconfig.xml"
    config.write_text(LABEL_CONFIG, encoding="utf8")

    start = annot / "label_studio_starten.sh"
    start.write_text(STARTSKRIPT.format(
        kachelwurzel=kachelwurzel, datadir=annot / "ls_data"), encoding="utf8")
    start.chmod(0o755)

    print(ANLEITUNG.format(
        config=config, kachelordner=kachelordner,
        export=annot / "export.json"))


if __name__ == "__main__":
    main()
