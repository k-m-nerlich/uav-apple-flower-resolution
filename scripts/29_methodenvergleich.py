#!/usr/bin/env python3
"""Schritt 29: Verfahren paarweise vergleichen (gleiche Kacheln, gleiche Falten).

Liest die Kachelzaehlungen, die 21_kv_auswertung.py je Verfahren schreibt
(<kv><suffix>_kacheln.csv, r = 12 px), und die der Farbschwelle (23, Mac). Je
Stufe wird F1 ueber alle Seeds gepoolt und die Differenz zweier Verfahren per
gepaartem Cluster-Bootstrap (dieselben Kachelgruppen wie in 21, 2000 Ziehungen)
mit 95-%-Intervall angegeben.

    python 29_methodenvergleich.py [--ign]

--ign (Version 2): Kacheldateien mit Ignorierzonen um 'unsicher'-Punkte
(<name>_ign_kacheln.csv aus 21 bzw. 23), Ausgabe methodenvergleich_ign.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
VERFAHREN = {                       # Name -> Kacheldatei aus 21
    "YOLOv8n + Punkt-NMS": "kv_bereinigt_pnms12_kacheln.csv",
    "DINOv2-s": "kv_ssl_dinov2_s_kacheln.csv",
    "I-JEPA H/14": "kv_ssl_ijepa_h14_1k_kacheln.csv",
    "I-JEPA H/14 x1.5": "kv_ssl_ijepa_h14_1k_x1.5_kacheln.csv",
    "YOLOv8n roh": "kv_bereinigt_kacheln.csv",
    "YOLOv8n gr roh": "kv_bereinigt_gr_kacheln.csv",
    "YOLOv8n gr + Punkt-NMS": "kv_bereinigt_gr_pnms12_kacheln.csv",
    "DINOv2-s gr": "kv_ssl_dinov2_s_gr_kacheln.csv",
    # Version 2 (Review 3.6, 3.7)
    "YOLOv8n NMS-IoU val": "kv_bereinigt_iouval_kacheln.csv",
    "YOLOv8s + Punkt-NMS": "kv_bereinigt_v8s_pnms12_kacheln.csv",
    "DINOv2-s x1.5": "kv_ssl_dinov2_s_x1.5_kacheln.csv",
}
VERGLEICHE = [("DINOv2-s", "YOLOv8n + Punkt-NMS"), ("I-JEPA H/14", "YOLOv8n + Punkt-NMS"),
              ("I-JEPA H/14 x1.5", "YOLOv8n + Punkt-NMS"), ("DINOv2-s", "I-JEPA H/14"),
              ("DINOv2-s", "I-JEPA H/14 x1.5"), ("I-JEPA H/14 x1.5", "I-JEPA H/14"),
              ("YOLOv8n + Punkt-NMS", "Farbschwelle"),
              ("YOLOv8n gr roh", "YOLOv8n roh"), ("YOLOv8n gr + Punkt-NMS", "YOLOv8n + Punkt-NMS"),
              ("DINOv2-s gr", "DINOv2-s"),
              ("YOLOv8n NMS-IoU val", "YOLOv8n + Punkt-NMS"), ("YOLOv8s + Punkt-NMS", "YOLOv8n + Punkt-NMS"),
              ("DINOv2-s x1.5", "DINOv2-s"), ("DINOv2-s x1.5", "YOLOv8n + Punkt-NMS"),
              ("DINOv2-s", "Farbschwelle")]


def lade(datei: str) -> dict:
    """stufe -> kachel -> [TP, FP, FN] (Summe ueber Seeds), nur r = 12 px."""
    aus = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    with open(ERG / datei, encoding="utf8") as f:
        for z in csv.DictReader(f):
            if float(z["radius_px"]) == 12.0:
                aus[z["stufe"]][z["kachel"]] += [int(z["TP"]), int(z["FP"]), int(z["FN"])]
    return aus


def lade_farbschwelle(sx: str = "") -> dict | None:
    pfad = ERG / f"schwellwert_kv{sx}_kacheln.csv"
    if not pfad.exists():
        return None
    with open(pfad, encoding="utf8") as f:
        zeilen = list(csv.DictReader(f))
    spalten = set(zeilen[0])
    if not {"stufe", "kachel", "TP", "FP", "FN"} <= spalten:
        print(f"Farbschwelle: Spalten {sorted(spalten)} nicht erkannt, uebersprungen")
        return None
    aus = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    for z in zeilen:
        if "radius_px" in z and float(z["radius_px"]) != 12.0:
            continue
        aus[z["stufe"]][z["kachel"]] += [int(z["TP"]), int(z["FP"]), int(z["FN"])]
    return aus


def f1(s):
    return 2 * s[..., 0] / (2 * s[..., 0] + s[..., 1] + s[..., 2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ign", action="store_true")
    sx = "_ign" if p.parse_args().ign else ""
    dateien = {n: d.replace("_kacheln.csv", f"{sx}_kacheln.csv") for n, d in VERFAHREN.items()}
    daten = {n: lade(d) for n, d in dateien.items() if (ERG / d).exists()}
    fs = lade_farbschwelle(sx)
    if fs:
        daten["Farbschwelle"] = fs
    gruppen = json.loads((ERG / "kv" / "kachelgruppen.json").read_text(encoding="utf8"))["gruppen"]
    stufen = sorted(next(iter(daten.values())))
    rng = np.random.default_rng(20261020)
    ziehungen = [rng.integers(0, len(gruppen), len(gruppen)) for _ in range(2000)]

    print(f"{'Verfahren':<22}" + "".join(f"{s:>9}" for s in stufen))
    for n, d in daten.items():
        print(f"{n:<22}" + "".join(f"{f1(sum(d[s].values())):9.3f}" for s in stufen))

    ergebnis = []
    print("\nDifferenz F1 (95-%-KI, gepaarter Cluster-Bootstrap ueber Kachelgruppen)")
    for a, b in VERGLEICHE:
        if a not in daten or b not in daten:
            continue
        zeile = f"{a + ' - ' + b:<42}"
        for s in stufen:
            kacheln = [k for g in gruppen for k in g]
            A = np.array([[daten[a][s][k] for k in g] for g in gruppen], dtype=object)
            B = np.array([[daten[b][s][k] for k in g] for g in gruppen], dtype=object)
            sa = np.array([np.sum(x, axis=0) for x in A]); sb = np.array([np.sum(x, axis=0) for x in B])
            punkt = f1(sa.sum(0)) - f1(sb.sum(0))
            boot = np.array([f1(sa[z].sum(0)) - f1(sb[z].sum(0)) for z in ziehungen])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            ergebnis.append({"a": a, "b": b, "stufe": s, "differenz": round(float(punkt), 4),
                             "ki95_lo": round(float(lo), 4), "ki95_hi": round(float(hi), 4)})
            zeile += f"  {s}: {punkt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(zeile)
    with open(ERG / f"methodenvergleich{sx}.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(ergebnis[0]))
        w.writeheader(); w.writerows(ergebnis)
    print(f"\nAusgabe: {ERG / f'methodenvergleich{sx}.csv'}")


if __name__ == "__main__":
    main()
