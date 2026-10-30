#!/usr/bin/env python3
"""Schritt 5: Kacheln fuer die Messung der MENSCHLICHEN DECKE erzeugen.

Frage
-----
Wie viele Blueten erkennt ein Mensch bei welcher Bodenaufloesung noch? Diese
Kurve braucht kein Modell, keine GPU und keine Trainingsdaten - sie misst, was
in den Bildern ueberhaupt sichtbar ist. Damit ist sie unabhaengig von allen
Schwaechen des Detektors und die belastbarste Zahl fuer den Abstract.

Verfahren
---------
Dieselben Kacheln werden auf jede Zielaufloesung heruntergerechnet und danach
wieder auf die urspruengliche Pixelgroesse gebracht. Die Information ist dann
weg, die Darstellungsgroesse aber gleich - sonst wuerde man messen, wie gut
jemand auf kleinen Bildern klickt, statt was er erkennen kann.

Blindheit
---------
Wer dieselbe Kachel mehrfach annotiert, erinnert sich, wo die Blueten waren.
Dagegen drei Massnahmen:
  * Die Ausgabedateien heissen k0001.jpg, k0002.jpg, ... Die Stufe ist dem
    Namen nicht anzusehen.
  * Die Reihenfolge ist durchmischt, Varianten derselben Kachel liegen weit
    auseinander.
  * Die Zuordnung steht ausschliesslich in schluessel.csv - waehrend des
    Annotierens NICHT hineinsehen.

Der Erinnerungseffekt laesst sich so daempfen, aber nicht beseitigen. Er
schoent die groben Stufen (man markiert, was man zu wissen glaubt) und macht
die Kurve damit flacher als die Wirklichkeit. Das gehoert als Limitation ins
Paper.

Die native Stufe ist bewusst noch einmal dabei: Ihr Vergleich mit der ersten
Annotationsrunde liefert die Intra-Rater-Uebereinstimmung gratis mit.

    python3 05_mehrstufen_kacheln.py --kacheln 12
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from gemeinsam import mm_pro_pixel, punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent

# Die real geflogenen Nadir-Hoehen in Ablass. 5.1 m ist die Referenzstufe und
# zugleich der Wiederholungstest fuer die Intra-Rater-Uebereinstimmung.
STUFEN_M = [5.1, 8.8, 10.7, 14.8, 16.7]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path,
                   default=PROJEKT / "annotation" / "export.json",
                   help="Referenzannotation der nativen Stufe")
    p.add_argument("--kachelordner", type=Path, default=None)
    p.add_argument("--ausgabe", type=Path,
                   default=PROJEKT / "kacheln_mehrstufig")
    p.add_argument("--kacheln", type=int, default=12,
                   help="Anzahl Kacheln; Aufgaben = Kacheln x Stufen")
    p.add_argument("--basis-hoehe", type=float, default=5.1)
    p.add_argument("--kronenhoehe", type=float, default=0.0)
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--stufen", type=float, nargs="+", default=STUFEN_M)
    p.add_argument("--seed", type=int, default=20260813)
    args = p.parse_args()

    if not args.export.exists():
        raise SystemExit(f"Referenzannotation nicht gefunden: {args.export}")

    kachelordner = args.kachelordner
    if kachelordner is None:
        unter = [d for d in sorted((PROJEKT / "kacheln").glob("*")) if d.is_dir()]
        if not unter:
            raise SystemExit("Keine Kacheln gefunden - erst 01_kacheln.py laufen lassen.")
        kachelordner = unter[0]

    # nur Kacheln verwenden, fuer die eine Referenzannotation existiert
    referenz = punkte_aus_export(args.export)
    verfuegbar = sorted(n for n in referenz
                        if (kachelordner / n).exists() and referenz[n]["punkte"])
    if not verfuegbar:
        raise SystemExit(f"Keine annotierte Kachel in {kachelordner} gefunden.")

    rng = random.Random(args.seed)
    gewaehlt = sorted(rng.sample(verfuegbar, min(args.kacheln, len(verfuegbar))))

    basis_gsd = mm_pro_pixel(args.basis_hoehe, args.kronenhoehe)
    je_kachel = {}
    for name in gewaehlt:
        varianten = [(name, h, mm_pro_pixel(h, args.kronenhoehe),
                      basis_gsd / mm_pro_pixel(h, args.kronenhoehe))
                     for h in args.stufen]
        rng.shuffle(varianten)
        je_kachel[name] = varianten

    # Rundenverteilung statt reinem Shuffle: In jeder Runde kommt hoechstens
    # eine Variante pro Kachel dran. Damit liegen die Varianten derselben
    # Kachel rund so weit auseinander, wie es Kacheln gibt - beim reinen
    # Shuffle landen sie regelmaessig nebeneinander, und dann erinnert man sich
    # beim zweiten Sehen an die Bluetenpositionen.
    aufgaben = []
    for runde in range(max(len(v) for v in je_kachel.values())):
        namen = [n for n in gewaehlt if runde < len(je_kachel[n])]
        rng.shuffle(namen)
        aufgaben.extend(je_kachel[n][runde] for n in namen)

    ziel = args.ausgabe
    ziel.mkdir(parents=True, exist_ok=True)
    for alt in ziel.glob("k*.jpg"):
        alt.unlink()

    zeilen = []
    for i, (name, hoehe, gsd, faktor) in enumerate(aufgaben, start=1):
        bild = Image.open(kachelordner / name).convert("RGB")
        anonym = f"k{i:04d}.jpg"
        if faktor >= 0.999:
            degradiert = bild                      # native Stufe unveraendert
            klein = args.kachel_px
        else:
            klein = max(32, int(round(args.kachel_px * faktor)))
            degradiert = (bild.resize((klein, klein), Image.LANCZOS)
                              .resize((args.kachel_px, args.kachel_px), Image.BICUBIC))
        degradiert.save(ziel / anonym, quality=95)
        zeilen.append({
            "anonym": anonym, "kachel": name,
            "hoehe_m": hoehe, "mm_pro_px": round(gsd, 2),
            "faktor": round(faktor, 4), "px_effektiv": klein,
            "referenzpunkte": len(referenz[name]["punkte"]),
        })

    schluessel = ziel / "schluessel.csv"
    with open(schluessel, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
        w.writeheader()
        w.writerows(zeilen)

    # Kontrolle: wie dicht liegen Varianten derselben Kachel beieinander?
    positionen = {}
    for i, z in enumerate(zeilen):
        positionen.setdefault(z["kachel"], []).append(i)
    abstaende = [b - a for p in positionen.values()
                 for a, b in zip(sorted(p), sorted(p)[1:])]

    print(f"Kacheln   : {len(gewaehlt)} von {len(verfuegbar)} annotierten")
    if abstaende:
        print(f"Blindheit : Varianten derselben Kachel liegen min. "
              f"{min(abstaende)}, im Median {sorted(abstaende)[len(abstaende)//2]} "
              f"Aufgaben auseinander")
    print(f"Stufen    : {', '.join(f'{h:.1f} m ({mm_pro_pixel(h, args.kronenhoehe):.1f} mm/px)' for h in args.stufen)}")
    print(f"Aufgaben  : {len(aufgaben)} -> {ziel}")
    print(f"Schluessel: {schluessel}   << waehrend des Annotierens NICHT oeffnen")
    print(f"\nGeschaetzte Annotationszeit: {len(aufgaben) * 4 // 60} bis "
          f"{len(aufgaben) * 8 // 60} Stunden.")
    print(f"""
Naechste Schritte
-----------------
1.  In Label Studio ein NEUES Projekt anlegen, z.B. "Menschliche Decke".
    Dieselbe Labeling-Konfiguration wie beim ersten Projekt verwenden
    ({PROJEKT / 'annotation' / 'labelconfig.xml'}).

2.  Die {len(aufgaben)} Dateien k*.jpg aus {ziel} hochladen.
    schluessel.csv NICHT hochladen.

3.  Annotieren - mit denselben Regeln wie in Runde 1. Wichtig: Bei groben
    Stufen NICHT laenger suchen als bei feinen. Gemessen wird, was auf einen
    normalen Blick erkennbar ist, nicht was mit maximalem Aufwand noch
    herausklaubbar waere. Zweifelsfall bleibt "unsicher".

4.  Am besten ueber mehrere Sitzungen verteilen, das daempft den
    Erinnerungseffekt zusaetzlich.

5.  Export als JSON nach
        {PROJEKT / 'annotation' / 'export_mehrstufig.json'}
    dann:
        python3 06_menschliche_decke.py
""")


if __name__ == "__main__":
    main()
