#!/usr/bin/env python3
"""Schritt 35: Erweiterter Aufloesungsbereich (Version 2, Review 7.2).

Vergleicht jede Stufe (1,71-20 mm/px) mit der nativen Stufe desselben
Verfahrens: F1-Differenz, gepaart ueber dieselben Testkacheln, Cluster-
Bootstrap ueber die 16 Kachelgruppen (wie 21/29), 95- und 90-%-KI und
Aequivalenz (90-%-KI innerhalb +-0,05). Liest die Kacheldateien aus 21
(r = 12 px, Ignorierzonen):
    YOLOv8n + Punkt-NMS  kv_bereinigt_pnms12_ign + kv_bereinigt_erw_pnms12_ign
    DINOv2-small         kv_ssl_dinov2_s_ign + kv_ssl_dinov2_s_erw_ign

Ausgabe: ergebnisse/erweiterter_bereich.csv

    python 35_erweiterter_bereich.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
VERFAHREN = {"YOLOv8n + Punkt-NMS": ("kv_bereinigt_pnms12_ign", "kv_bereinigt_erw_pnms12_ign"),
             "DINOv2-s": ("kv_ssl_dinov2_s_ign", "kv_ssl_dinov2_s_erw_ign"),
             "DINOv2-s x1.5": ("kv_ssl_dinov2_s_x1.5_ign", None)}
MARGE = 0.05


def lade(name: str) -> tuple[dict, dict]:
    """stufe -> kachel -> [TP, FP, FN] (Summe ueber Seeds, r = 12), stufe -> GSD."""
    aus, gsd = defaultdict(lambda: defaultdict(lambda: np.zeros(3))), {}
    with open(ERG / f"{name}_kacheln.csv", encoding="utf8") as f:
        for z in csv.DictReader(f):
            if float(z["radius_px"]) == 12.0:
                aus[z["stufe"]][z["kachel"]] += [int(z["TP"]), int(z["FP"]), int(z["FN"])]
    with open(ERG / f"{name}_stufen.csv", encoding="utf8") as f:
        gsd.update({z["stufe"]: float(z["gsd_gemessen"]) for z in csv.DictReader(f)})
    return aus, gsd


def f1(s):
    return 2 * s[..., 0] / (2 * s[..., 0] + s[..., 1] + s[..., 2])


def main() -> None:
    gruppen = json.loads((ERG / "kv" / "kachelgruppen.json").read_text(encoding="utf8"))["gruppen"]
    rng = np.random.default_rng(20261022)
    ziehungen = [rng.integers(0, len(gruppen), len(gruppen)) for _ in range(2000)]
    zeilen = []
    for name, dateien in VERFAHREN.items():
        daten, gsd = {}, {}
        for d in dateien:
            if d and (ERG / f"{d}_kacheln.csv").exists():
                a, g = lade(d); daten.update(a); gsd.update(g)
        stufen = sorted(daten, key=gsd.get)
        je = {s: np.array([sum((daten[s][k] for k in g), np.zeros(3)) for g in gruppen]) for s in stufen}
        nativ = stufen[0]
        for s in stufen:
            d = f1(je[s].sum(0)) - f1(je[nativ].sum(0))
            boot = np.array([f1(je[s][z].sum(0)) - f1(je[nativ][z].sum(0)) for z in ziehungen])
            k95, k90 = np.quantile(boot, [.025, .975]), np.quantile(boot, [.05, .95])
            zeilen.append({"verfahren": name, "stufe": s, "gsd": gsd[s], "f1": round(float(f1(je[s].sum(0))), 4),
                           "diff_nativ": round(float(d), 4), "ki95_lo": round(float(k95[0]), 4),
                           "ki95_hi": round(float(k95[1]), 4), "ki90_lo": round(float(k90[0]), 4),
                           "ki90_hi": round(float(k90[1]), 4),
                           "aequivalent": bool(-MARGE < k90[0] and k90[1] < MARGE)})
            z = zeilen[-1]
            print(f"{name:<20} {gsd[s]:6.2f} mm/px  F1 {z['f1']:.3f}  Diff {d:+.3f} "
                  f"[{k95[0]:+.3f}; {k95[1]:+.3f}]  aequivalent: {z['aequivalent']}")
    with open(ERG / "erweiterter_bereich.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0])); w.writeheader(); w.writerows(zeilen)
    print(f"-> {ERG / 'erweiterter_bereich.csv'}")


if __name__ == "__main__":
    main()
