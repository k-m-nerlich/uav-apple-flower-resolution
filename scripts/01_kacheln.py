#!/usr/bin/env python3
"""Schritt 1: Kacheln aus einem Flug extrahieren, zentriert auf die Baumreihe.

Warum reihenzentriert: Die Reihen in Ablass sind schmal, ein grosser Teil jedes
Bildes ist Fahrgasse und Boden. Zufaellige oder mittige Kacheln sind zur Haelfte
leer - das verschwendet Annotationszeit und verzerrt die Klassenverteilung.

Aufruf (Standardwerte passen auf den feinsten Nadir-Flug in Ablass):

    python3 01_kacheln.py --anzahl 25

Ausgabe:
    kacheln/<flugname>/*.jpg        die Kacheln
    kacheln/<flugname>/manifest.csv Herkunft, Hoehe, GSD, Schaerfe, Scores
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from gemeinsam import (ssd_wurzel, bluetenscore, finde_bilder, lies_xmp, mm_pro_pixel,
                       reihenkandidaten, schaerfe)

SSD = ssd_wurzel()
STANDARD_FLUG = SSD / "Ablass Elstar" / "DJI_202604291329_009_UgCS-Missionof4-23-2026441PMBEOLGSD0,1"
PROJEKT = Path(__file__).resolve().parent.parent


def kandidaten(bild: Image.Image, kachel_px: int, variante: str,
               auswahl: str = "reihen", schritt_teiler: int = 3):
    """Liefert bewertete Fensterpositionen als (merkmale, x, y), bestbewertet zuerst.

    auswahl='reihen'  Reihengeometrie schaetzen und entlang der Mittellinien
                      abtasten (siehe gemeinsam.reihenkandidaten). Der
                      Bluetenscore dient dann nur noch der Rangfolge - er muss
                      die Reihe nicht mehr finden, sondern nur noch zwischen
                      echten Reihen und faelschlich erkannten Grasgassen
                      unterscheiden.
    auswahl='raster'  Vollstaendiges Abrastern, nur Bluetenscore. Faellt
                      haeufiger auf abgetrockneten Boden herein, braucht aber
                      keine Reihenstruktur - fuer Streuobst oder Junganlagen
                      ohne klare Reihen die robustere Wahl.
    """
    breite, hoehe = bild.size
    klein = np.asarray(bild.resize((breite // 8, hoehe // 8)))
    k8 = max(4, kachel_px // 8)

    if auswahl == "reihen":
        zentren, _grad, _n = reihenkandidaten(bild, kachel_px)
        positionen = [(cx - kachel_px // 2, cy - kachel_px // 2)
                      for cx, cy in zentren]
    else:
        schritt = max(1, k8 // schritt_teiler)
        positionen = [(x * 8, y * 8)
                      for y in range(0, klein.shape[0] - k8 + 1, schritt)
                      for x in range(0, klein.shape[1] - k8 + 1, schritt)]

    treffer = []
    for x, y in positionen:
        x8, y8 = max(0, x // 8), max(0, y // 8)
        fenster = klein[y8:y8 + k8, x8:x8 + k8]
        if fenster.shape[0] < k8 // 2 or fenster.shape[1] < k8 // 2:
            continue
        treffer.append((bluetenscore(fenster, variante), x, y))

    treffer.sort(key=lambda t: t[0]["score"], reverse=True)
    return treffer


def ohne_ueberlappung(treffer, kachel_px, maximal):
    """Greedy-Auswahl: bester Treffer zuerst, dann nur noch disjunkte Fenster."""
    gewaehlt = []
    for m, x, y in treffer:
        if all(abs(x - gx) >= kachel_px or abs(y - gy) >= kachel_px
               for _, gx, gy in gewaehlt):
            gewaehlt.append((m, x, y))
        if len(gewaehlt) >= maximal:
            break
    return gewaehlt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flug", type=Path, default=STANDARD_FLUG)
    p.add_argument("--anzahl", type=int, default=25, help="Gesamtzahl Kacheln")
    p.add_argument("--kachel-px", type=int, default=1024,
                   help="Kantenlaenge der AUSGEGEBENEN Kachel in Pixeln")
    p.add_argument("--boden-m", type=float, default=None,
                   help="Zielabdeckung am Boden in Metern. Ist der Wert gesetzt, "
                        "wird der native Ausschnitt aus der Flughoehe berechnet "
                        "und anschliessend auf --kachel-px skaliert. Damit sind "
                        "Kacheln aus Fluegen unterschiedlicher Hoehe direkt "
                        "vergleichbar - sonst zeigt eine 1024er Kachel je nach "
                        "Hoehe voellig verschiedene Bodenflaechen.")
    p.add_argument("--pro-bild", type=int, default=2)
    p.add_argument("--ueberzeichnung", type=float, default=2.5,
                   help="Faktor, um den mehr Kandidaten gesammelt werden, als "
                        "am Ende behalten werden")
    p.add_argument("--kronenhoehe", type=float, default=0.0,
                   help="Baumhoehe in m; korrigiert die Aufnahmedistanz")
    p.add_argument("--min-schaerfe", type=float, default=None,
                   help="Absolute Schwelle. Ohne Angabe wird das unterste "
                        "--unschaerfe-quantil der Verteilung verworfen.")
    p.add_argument("--unschaerfe-quantil", type=float, default=0.15,
                   help="Anteil der unschaerfsten Kacheln, der verworfen wird")
    p.add_argument("--auswahl", default="reihen", choices=["reihen", "raster"],
                   help="reihen = Reihengeometrie schaetzen (Standard), "
                        "raster = vollstaendig abrastern")
    p.add_argument("--variante", default="weiss_textur",
                   choices=["weiss_textur", "weiss_schatten", "veg_weiss", "kombiniert"],
                   help="Formel fuer die Reihenerkennung, siehe gemeinsam.bluetenscore")
    p.add_argument("--ausgabe", type=Path, default=None)
    args = p.parse_args()

    if not args.flug.is_dir():
        raise SystemExit(f"Flugordner nicht gefunden: {args.flug}\n"
                         "Ist die SSD gemountet?")

    ziel = args.ausgabe or (PROJEKT / "kacheln" / args.flug.name[:40])
    ziel.mkdir(parents=True, exist_ok=True)

    bilder = finde_bilder(args.flug)
    if not bilder:
        raise SystemExit(f"Keine RGB-Bilder (_V) in {args.flug}")

    # Mehr Kandidaten sammeln als gebraucht, damit die Rangfolge nach Score
    # tatsaechlich aussieben kann. Gleichmaessig ueber den Flug verteilt, damit
    # nicht nur ein Reihenende in der Stichprobe landet.
    noetig = max(1, -(-int(args.anzahl * args.ueberzeichnung) // args.pro_bild))
    schritt = max(1, len(bilder) // noetig)
    auswahl = bilder[::schritt][:noetig]

    print(f"Flug      : {args.flug.name}")
    print(f"Bilder    : {len(bilder)} vorhanden, {len(auswahl)} beprobt")

    # Erst alle Kandidaten sammeln, dann nach Schaerfe filtern. Grund: eine
    # absolute Schaerfeschwelle laesst sich vorab nicht sinnvoll setzen, die
    # Laplace-Varianz haengt stark vom Bildinhalt ab (gemessen: 1900-3300).
    kandidatenliste = []
    for bildpfad in auswahl:
        xmp = lies_xmp(bildpfad)
        hoehe = xmp.get("RelativeAltitude", float("nan"))
        gsd = mm_pro_pixel(hoehe, args.kronenhoehe)

        # Schnittgroesse im Originalbild: entweder fest in Pixeln oder aus der
        # gewuenschten Bodenabdeckung und der Flughoehe abgeleitet.
        if args.boden_m:
            schnitt_px = max(64, int(round(args.boden_m * 1000.0 / gsd)))
        else:
            schnitt_px = args.kachel_px

        bild = Image.open(bildpfad).convert("RGB")
        if schnitt_px >= min(bild.size):
            schnitt_px = min(bild.size) - 2

        gewaehlt = ohne_ueberlappung(
            kandidaten(bild, schnitt_px, args.variante, args.auswahl),
            schnitt_px, args.pro_bild)

        for m, x, y in gewaehlt:
            kachel = bild.crop((x, y, x + schnitt_px, y + schnitt_px))
            if schnitt_px != args.kachel_px:
                verfahren = (Image.LANCZOS if schnitt_px > args.kachel_px
                             else Image.BICUBIC)
                kachel = kachel.resize((args.kachel_px, args.kachel_px), verfahren)
            kandidatenliste.append({
                "kachel": f"{bildpfad.stem}_x{x}_y{y}.jpg",
                "quellbild": bildpfad.name,
                "x": x, "y": y, "kachel_px": args.kachel_px,
                "schnitt_px": schnitt_px,
                "boden_m": round(schnitt_px * gsd / 1000.0, 3),
                "hoehe_ueber_start_m": round(hoehe, 2),
                "gimbal_pitch": xmp.get("GimbalPitchDegree", ""),
                "mm_pro_px": round(gsd, 3),
                # Achtung: Nach dem Hochskalieren hat die Kachel zwar
                # args.kachel_px Pixel, traegt aber nur die Information von
                # schnitt_px Pixeln. Massgeblich fuer jede Aussage zur
                # Aufloesung ist mm_pro_px (nativ), NICHT die Pixelzahl der
                # Kachel.
                "mm_pro_px_darstellung": round(
                    schnitt_px * gsd / args.kachel_px, 3),
                "schaerfe": round(schaerfe(np.asarray(kachel.convert("L"))), 1),
                "score": round(m["score"], 6),
                "vegetationsanteil": round(m["vegetation"], 4),
                "weissanteil": round(m["weiss"], 4),
                "schattenanteil": round(m["schatten"], 4),
                "textur": round(m["textur"], 4),
                "_bild": kachel,
            })

    if not kandidatenliste:
        raise SystemExit("Keine Kandidaten gefunden.")

    werte = sorted(k["schaerfe"] for k in kandidatenliste)
    if args.min_schaerfe is not None:
        schwelle = args.min_schaerfe
    else:
        idx = int(len(werte) * args.unschaerfe_quantil)
        schwelle = werte[min(idx, len(werte) - 1)]

    scharf = [k for k in kandidatenliste if k["schaerfe"] >= schwelle]
    verworfen_unscharf = len(kandidatenliste) - len(scharf)
    scharf.sort(key=lambda k: k["score"], reverse=True)

    zeilen = []
    for k in scharf[:args.anzahl]:
        k.pop("_bild").save(ziel / k["kachel"], quality=95)
        zeilen.append(k)

    if not zeilen:
        raise SystemExit("Keine Kachel hat die Schaerfeschwelle bestanden.")

    manifest = ziel / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
        w.writeheader()
        w.writerows(zeilen)

    schaerfen = [z["schaerfe"] for z in zeilen]
    print(f"Auswahl   : '{args.auswahl}' + Score '{args.variante}', "
          f"{len(kandidatenliste)} Kandidaten")
    print(f"Kacheln   : {len(zeilen)} geschrieben nach {ziel}")
    print(f"            {verworfen_unscharf} wegen Unschaerfe verworfen "
          f"(Schwelle {schwelle:.0f})")
    print(f"GSD       : {zeilen[0]['mm_pro_px']} mm/px nativ "
          f"<- massgeblich fuer die Informationsdichte")
    print(f"            {zeilen[0]['mm_pro_px_darstellung']} mm/px nur "
          f"rechnerisch nach dem Hochskalieren")
    print(f"Abdeckung : {zeilen[0]['boden_m']} m Boden je Kachel "
          f"(Schnitt {zeilen[0]['schnitt_px']} px -> {args.kachel_px} px)")
    print(f"Schaerfe  : min {min(schaerfen):.0f} / median "
          f"{sorted(schaerfen)[len(schaerfen)//2]:.0f} / max {max(schaerfen):.0f}")
    print(f"Manifest  : {manifest}")
    print("\nNaechster Schritt: Kacheln sichten. Sitzen sie auf der Reihe?")
    print("Falls nicht, --pro-bild erhoehen oder bluetenscore() in gemeinsam.py anpassen.")


if __name__ == "__main__":
    main()
