#!/usr/bin/env python3
"""Schritt 13: Die Bodenaufloesung empirisch messen statt sie anzunehmen.

Warum
-----
Alle bisherigen GSD-Angaben beruhen auf zwei ungeprueften Annahmen:
Bodenbreite = 1.50 * Aufnahmedistanz (24 mm KB-Aequivalent) und
Aufnahmedistanz = RelativeAltitude, also Hoehe ueber STARTPUNKT statt ueber
Kronendach. Bei 2-3 m Baumhoehe und 4,7 m Flughoehe ist der Unterschied ein
Faktor 2. Der Versuch, einen Hoehenstapel zu registrieren, ist genau daran
gescheitert: Nach Vorkorrektur mit dem angenommenen Modell existierte keine
konsistente Restverschiebung mehr.

Verfahren
---------
Aufeinanderfolgende Bilder EINES Fluges ueberlappen stark. Zwischen ihnen
gilt: gleiche Hoehe, gleiche Blickrichtung, kein Massstabsunterschied, keine
Parallaxeaenderung. Ihre Verschiebung laesst sich sauber messen - in Pixeln
per SIFT, in Metern aus den GPS-Koordinaten. Der Quotient ist die
Bodenaufloesung, ohne jede Annahme ueber Brennweite oder Sensorgroesse.

Ueber mehrere Fluege verschiedener Hoehe ergibt sich GSD(Hoehe). Eine Gerade
durch diese Punkte liefert beides: die tatsaechliche Brennweitenkonstante
(Steigung) und den Hoehenbezug (Nullstelle) - also, auf welche Ebene sich
RelativeAltitude wirklich bezieht.

    python3 13_geometrie_kalibrieren.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gemeinsam import ssd_wurzel, lies_gps, lies_xmp, meter_pro_grad

PROJEKT = Path(__file__).resolve().parent.parent
SSD = ssd_wurzel()


def verschiebung_px(a: Path, b: Path, max_px: float = 2500.0):
    """Verschiebung zwischen zwei Aufnahmen in Pixeln, per SIFT + Dichtekern."""
    import cv2
    sift = cv2.SIFT_create(nfeatures=3000)
    ga = cv2.createCLAHE(2.0, (8, 8)).apply(np.asarray(Image.open(a).convert("L")))
    gb = cv2.createCLAHE(2.0, (8, 8)).apply(np.asarray(Image.open(b).convert("L")))
    k1, d1 = sift.detectAndCompute(ga, None)
    k2, d2 = sift.detectAndCompute(gb, None)
    if d1 is None or d2 is None:
        return None, 0
    paare = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    gut = [m for m, n in (p for p in paare if len(p) == 2)
           if m.distance < 0.75 * n.distance]
    if len(gut) < 15:
        return None, len(gut)
    v = np.array([[k1[m.trainIdx].pt[0] - k2[m.queryIdx].pt[0],
                   k1[m.trainIdx].pt[1] - k2[m.queryIdx].pt[1]] for m in gut])
    v = v[np.hypot(v[:, 0], v[:, 1]) <= max_px]
    if len(v) < 15:
        return None, len(gut)
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    n = (d <= 8.0).sum(1)
    if n.max() < 10:
        return None, int(n.max())          # kein tragfaehiger Konsens
    kern = v[n >= n.max() * 0.6]
    if len(kern) == 0:
        return None, int(n.max())          # sonst liefert np.median NaN
    return np.median(kern, axis=0), int(n.max())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path,
                   default=PROJEKT / "ergebnisse" / "bildindex.csv")
    p.add_argument("--wurzel", type=Path, default=SSD)
    p.add_argument("--paare", type=int, default=12, help="Paare je Flug")
    p.add_argument("--min-m", type=float, default=1.0,
                   help="Mindestabstand eines Paars in Metern")
    p.add_argument("--max-m", type=float, default=12.0)
    args = p.parse_args()

    index = list(csv.DictReader(open(args.index, encoding="utf8")))
    fluege = defaultdict(list)
    for e in index:
        if float(e["pitch"]) <= -80:
            fluege[e["flug"]].append(e)

    print(f"{'Flug':<40}{'Hoehe':>7}{'Paare':>7}{'mm/px gemessen':>17}{'Streuung':>10}")
    punkte = []
    for flug, es in sorted(fluege.items()):
        es.sort(key=lambda e: e["pfad"])
        alt = float(np.median([float(e["alt"]) for e in es]))
        werte = []
        i = 0
        while i < len(es) - 1 and len(werte) < args.paare:
            a, b = es[i], es[i + 1]
            i += 1 if len(es) < 200 else max(1, len(es) // (args.paare * 4))
            m_b, m_l = meter_pro_grad(float(a["lat"]))
            dm = math.hypot((float(b["lat"]) - float(a["lat"])) * m_b,
                            (float(b["lon"]) - float(a["lon"])) * m_l)
            if not (args.min_m <= dm <= args.max_m):
                continue
            if abs(float(a["yaw"]) - float(b["yaw"])) > 8:
                continue
            v, n = verschiebung_px(args.wurzel / a["pfad"], args.wurzel / b["pfad"])
            if v is None:
                continue
            px = float(np.hypot(*v))
            if not np.isfinite(px) or px < 50:
                continue
            werte.append(dm / px * 1000.0)          # mm je Pixel

        if len(werte) >= 3:
            med = float(np.median(werte))
            iqr = float(np.percentile(werte, 75) - np.percentile(werte, 25))
            print(f"{flug[:39]:<40}{alt:6.1f}m{len(werte):7d}{med:17.3f}{iqr:10.3f}")
            punkte.append((alt, med, len(werte)))
        else:
            print(f"{flug[:39]:<40}{alt:6.1f}m{len(werte):7d}{'zu wenige Paare':>17}")

    if len(punkte) < 3:
        raise SystemExit("\nZu wenige Fluege ausgewertet fuer eine Anpassung.")

    h = np.array([p[0] for p in punkte])
    g = np.array([p[1] for p in punkte])
    steigung, achse = np.polyfit(h, g, 1)
    rest = g - (steigung * h + achse)
    r2 = 1 - (rest ** 2).sum() / ((g - g.mean()) ** 2).sum()
    nullstelle = -achse / steigung

    print(f"\nAnpassung GSD(mm/px) = {steigung:.4f} * Hoehe(m) + {achse:.4f}")
    print(f"  Bestimmtheitsmass R2 = {r2:.4f}")
    print(f"  Nullstelle bei {nullstelle:+.2f} m Hoehe ueber Startpunkt")
    print(f"\nZum Vergleich das bisher angenommene Modell:")
    print(f"  GSD = 1.50 * Hoehe * 1000 / 4000 = {1.5*1000/4000:.4f} * Hoehe")
    print(f"  gemessene Steigung / angenommene = "
          f"{steigung/(1.5*1000/4000):.3f}")
    if abs(nullstelle) > 0.5:
        print(f"\nDie Nullstelle liegt nicht bei 0. RelativeAltitude bezieht sich")
        print(f"also nicht auf die abgebildete Ebene; der Versatz betraegt")
        print(f"{nullstelle:+.2f} m - plausibel als Hoehe des Kronendachs ueber")
        print(f"dem Startpunkt.")

    ziel = PROJEKT / "ergebnisse" / "geometrie_kalibrierung.csv"
    with open(ziel, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["hoehe_m", "mm_pro_px_gemessen", "n_paare"])
        w.writerows([[round(a, 2), round(b, 4), c] for a, b, c in punkte])
    print(f"\nTabelle   : {ziel}")


if __name__ == "__main__":
    main()
