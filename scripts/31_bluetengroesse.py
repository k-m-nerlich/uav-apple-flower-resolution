#!/usr/bin/env python3
"""Schritt 31: Groesse je annotierter Bluete aus dem Bild schaetzen.

Hintergrund (Befund 7): Die Blueten sind je Kachel im Median 14 bis 47 mm
gross, die Modelle lernen aber eine feste Groesse (YOLO 24-px-Boxen, SSL-Kopf
Heatmap-Breite 5 px). Grosse Blueten werden dadurch mehrfach oder neben der
Mitte getroffen.

Verfahren
---------
Um jeden Punkt der Referenz A (Runde 1) wird die zusammenhaengende weisse
Flaeche gesucht (max(R,G,B) >= 200, max - min <= 40, 8er-Nachbarschaft), die
ein 13 x 13-px-Fenster um den Punkt beruehrt. Groesse = Aequivalentdurchmesser
2 * sqrt(Flaeche / pi). Flaechen ueber 3000 px (zusammengewachsene Bueschel,
Pfahl) zaehlen nicht.
Ohne Messung (Bluete im Schatten, kein weisser Fleck): Median der Kachel,
sonst Median aller Messungen. Ergebnis auf [MIN_PX, MAX_PX] begrenzt.

Ausgabe: annotation/punktgroessen.json
    {"meta": {...}, "groessen": {kachel: [durchmesser_px je Punkt, ...]}}
in derselben Reihenfolge wie gemeinsam.punkte_aus_export(export.json).

    python 31_bluetengroesse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gemeinsam import punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent
HELL, CHROMA, FENSTER, MAX_FLAECHE = 200, 40, 6, 3000
MIN_PX, MAX_PX = 12.0, 48.0
GSD = 1.714


def main() -> None:
    ordner = [d for d in sorted((PROJEKT / "kacheln").glob("*")) if d.is_dir()][0]
    roh = punkte_aus_export(PROJEKT / "annotation" / "export.json")
    gemessen, alle = {}, []
    for k, e in roh.items():
        if not (ordner / k).exists():
            continue
        img = np.asarray(Image.open(ordner / k).convert("RGB")).astype(np.int16)
        hell = img.max(2)
        maske = ((hell >= HELL) & ((hell - img.min(2)) <= CHROMA)).astype(np.uint8)
        _, lab, st, _ = cv2.connectedComponentsWithStats(maske, connectivity=8)
        werte = []
        for x, y in np.array(e["punkte"], float).reshape(-1, 2) * 10.24:
            x0, x1 = int(max(0, x - FENSTER)), int(min(1023, x + FENSTER))
            y0, y1 = int(max(0, y - FENSTER)), int(min(1023, y + FENSTER))
            ids = np.unique(lab[y0:y1 + 1, x0:x1 + 1]); ids = ids[ids > 0]
            fl = max((st[i, cv2.CC_STAT_AREA] for i in ids), default=0)
            werte.append(2 * np.sqrt(fl / np.pi) if 0 < fl < MAX_FLAECHE else np.nan)
        gemessen[k] = np.array(werte)
        alle += [w for w in werte if not np.isnan(w)]
    global_med = float(np.median(alle))
    groessen, n_ersatz = {}, 0
    for k, w in gemessen.items():
        med = float(np.nanmedian(w)) if np.isfinite(w).sum() >= 3 else global_med
        n_ersatz += int(np.isnan(w).sum())
        groessen[k] = [round(float(np.clip(v if np.isfinite(v) else med, MIN_PX, MAX_PX)), 1)
                       for v in w]
    flach = np.concatenate([np.array(v) for v in groessen.values()])
    meta = {"punkte": int(len(flach)), "gemessen": int(len(alle)), "ersetzt": n_ersatz,
            "median_px": float(np.median(flach)), "p10_px": float(np.percentile(flach, 10)),
            "p90_px": float(np.percentile(flach, 90)), "grenzen_px": [MIN_PX, MAX_PX],
            "kriterium": f"max>={HELL}, max-min<={CHROMA}, Fenster +-{FENSTER} px, Flaeche<{MAX_FLAECHE}"}
    (PROJEKT / "annotation" / "punktgroessen.json").write_text(
        json.dumps({"meta": meta, "groessen": groessen}, indent=1), encoding="utf8")
    print(f"{meta['punkte']} Punkte, {meta['gemessen']} gemessen, {n_ersatz} mit Kachelmedian ersetzt")
    print(f"Durchmesser nach Begrenzung: Median {meta['median_px']:.1f} px "
          f"({meta['median_px']*GSD:.0f} mm), P10 {meta['p10_px']:.1f}, P90 {meta['p90_px']:.1f} px")


if __name__ == "__main__":
    main()
