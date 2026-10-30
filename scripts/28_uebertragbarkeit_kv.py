#!/usr/bin/env python3
"""Schritt 25: Uebertragbarkeit auf Zauberwald, neu mit den KV-Modellen.

Was 12_uebertragbarkeit.py anders machte (und warum das ersetzt wird)
--------------------------------------------------------------------
* EIN Modell, Schwelle auf den Zauberwald-Kacheln selbst gesucht (Orakel)
* Radius 24 px auf den nativen Kacheln = 130 mm, in Ablass 12 px = 21 mm
* kein Massstabsabgleich: Die Modelle sehen Bilder im 1,714-mm/px-Raster
  (degradierter Inhalt, auf 1024 px zurueckgerechnet), Zauberwald-Kacheln
  liegen nativ bei 5,44 mm/px - eine Bluete erschien dem Modell ~3x kleiner.

Hier
----
* Zauberwald-Kacheln (1024 px, gemessen 5,44 mm/px) werden bikubisch auf das
  Trainingsraster gebracht (Faktor 5,44/1,714, auf ein Vielfaches von 32
  gerundet) - genau wie die degradierten Ablass-Stufen.
* Modelle der bereinigten KV der Stufe mit naechster GSD (h167, 5,61 mm/px;
  h148 als Gegenprobe), alle 6 Falten x 10 Seeds = 60 Modelle, jeweils mit der
  auf ihrer Ablass-Validierungsfalte gewaehlten Schwelle. Keine Anpassung
  an Zauberwald.
* Zuordnung in Millimetern wie in Ablass: 21 mm (= 12 px) und 27 mm (= 16 px),
  dazu 130 mm (= alter 24-px-Radius) zum Vergleich.
* In-domain-Vergleich: dieselben Modelle auf ihren Ablass-Testkacheln.

Ausgabe: ergebnisse/uebertragbarkeit_kv.csv, uebertragbarkeit_kv.json

    python 28_uebertragbarkeit_kv.py --device 0                  (richtig)
    python 28_uebertragbarkeit_kv.py --device 0 --raster nativ   (wie 12, Vergleich)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gemeinsam import punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
GSD_ABLASS_RASTER = 1.714
GSD_ZAUBERWALD = 5.44
RADIEN_MM = {"21mm": 12 * GSD_ABLASS_RASTER, "27mm": 16 * GSD_ABLASS_RASTER, "130mm": 24 * GSD_ZAUBERWALD}
SCHWELLEN_RADIUS = {"21mm": 12.0, "27mm": 16.0, "130mm": 12.0}   # welche val-Schwelle gilt


def zuordnen(w, v, r):
    if len(w) == 0:
        return 0, len(v), 0
    if len(v) == 0:
        return 0, 0, len(w)
    d = np.linalg.norm(w[:, None, :] - v[None, :, :], axis=2)
    fw, fv = np.ones(len(w), bool), np.ones(len(v), bool)
    tp = 0
    for i, j in sorted(np.argwhere(d <= r).tolist(), key=lambda p: d[p[0], p[1]]):
        if fw[i] and fv[j]:
            fw[i] = fv[j] = False
            tp += 1
    return tp, int(fv.sum()), int(fw.sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stufen", nargs="+", default=["h167", "h148"])
    p.add_argument("--device", default="mps")
    p.add_argument("--raster", choices=["training", "nativ"], default="training",
                   help="training: Kacheln auf das 1,714-mm/px-Raster hochrechnen (richtig); "
                        "nativ: Kacheln unveraendert wie in 12_uebertragbarkeit.py (Vergleich)")
    args = p.parse_args()
    endung = "" if args.raster == "training" else "_nativ"

    from ultralytics import YOLO

    roh = punkte_aus_export(PROJEKT / "annotation" / "export_zauberwald.json")
    kacheln = sorted(n for n in roh if (PROJEKT / "kacheln_zauberwald" / n).exists())
    ref = {k: np.array(roh[k]["punkte"], float).reshape(-1, 2) * 10.24 for k in kacheln}  # native px
    groesse = (int(round(1024 * GSD_ZAUBERWALD / GSD_ABLASS_RASTER / 32)) * 32
               if args.raster == "training" else 1024)
    faktor = groesse / 1024
    print(f"Zauberwald: {len(kacheln)} Kacheln, {sum(len(v) for v in ref.values())} Punkte; "
          f"hochgerechnet auf {groesse} px (Faktor {faktor:.3f}, Raster "
          f"{GSD_ZAUBERWALD/faktor:.3f} mm/px)")
    bilder = {k: Image.open(PROJEKT / "kacheln_zauberwald" / k).convert("RGB")
                    .resize((groesse, groesse), Image.BICUBIC) for k in kacheln}

    conf = {}
    for z in csv.DictReader(open(ERG / "kv_bereinigt_kacheln.csv", encoding="utf8")):
        conf[(z["stufe"], int(z["seed"]), int(z["falte"]), float(z["radius_px"]))] = float(z["conf"])

    zeilen, zusammen = [], {}
    for stufe in args.stufen:
        for gew in sorted((ERG / "kv_bereinigt" / "laeufe" / stufe).glob("f*_s*/weights/best.pt")):
            lauf = gew.parent.parent.name
            falte, seed = int(lauf.split("_")[0][1:]), int(lauf.split("_")[1][1:])
            modell = YOLO(str(gew))
            det = {}
            for k in kacheln:
                r = modell.predict(bilder[k], conf=0.01, imgsz=groesse, max_det=20000,
                                   device=args.device, verbose=False)[0]
                det[k] = (np.c_[r.boxes.xywh.cpu().numpy()[:, :2] / faktor,
                                r.boxes.conf.cpu().numpy()] if len(r.boxes) else np.zeros((0, 3)))
            for name, r_mm in RADIEN_MM.items():
                c = conf[(stufe, seed, falte, SCHWELLEN_RADIUS[name])]
                tp = fp = fn = 0; nd, nr = [], []
                for k in kacheln:
                    v = det[k][det[k][:, 2] >= c, :2]
                    a, b_, e = zuordnen(ref[k], v, r_mm / GSD_ZAUBERWALD)
                    tp += a; fp += b_; fn += e; nd.append(len(v)); nr.append(len(ref[k]))
                nd, nr = np.array(nd), np.array(nr)
                # Orakel: beste Schwelle auf den Zauberwald-Kacheln selbst (alte Praxis)
                orakel = 0.0
                for co in np.arange(0.05, 0.951, 0.05):
                    s = np.zeros(3)
                    for k in kacheln:
                        s += zuordnen(ref[k], det[k][det[k][:, 2] >= co, :2], r_mm / GSD_ZAUBERWALD)
                    orakel = max(orakel, 2 * s[0] / (2 * s[0] + s[1] + s[2]))
                zeilen.append({"stufe": stufe, "falte": falte, "seed": seed, "radius": name,
                               "conf": c, "f1_orakel": orakel, "TP": tp, "FP": fp, "FN": fn,
                               "precision": tp / (tp + fp) if tp + fp else 0.0,
                               "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
                               "det_je_ref": nd.sum() / nr.sum(),
                               "zaehl_r2": float(np.corrcoef(nd, nr)[0, 1] ** 2) if nd.std() else 0.0})
            print(f"{stufe} {lauf}: F1 21 mm {zeilen[-3]['f1']:.3f}, 27 mm {zeilen[-2]['f1']:.3f}, "
                  f"130 mm {zeilen[-1]['f1']:.3f}, Det/Ref {zeilen[-1]['det_je_ref']:.2f}", flush=True)

    with open(ERG / f"uebertragbarkeit_kv{endung}.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
        w.writeheader()
        w.writerows([{k: round(v, 4) if isinstance(v, float) else v for k, v in z.items()} for z in zeilen])

    # In-domain: dieselben Modelle auf ihren Ablass-Testkacheln (pro Seed ueber 36 Kacheln gepoolt)
    laeufe = list(csv.DictReader(open(ERG / "kv_bereinigt_laeufe.csv", encoding="utf8")))
    print("\nZusammenfassung (Mittel ueber Seeds; je Seed die 6 Faltenmodelle gemittelt)")
    for stufe in args.stufen:
        for name in RADIEN_MM:
            zz = [z for z in zeilen if z["stufe"] == stufe and z["radius"] == name]
            je_seed = defaultdict(list)
            for z in zz:
                je_seed[z["seed"]].append(z["f1"])
            s = np.array([np.mean(v) for v in je_seed.values()])
            fz = np.array([z["f1"] for z in zz])
            eintrag = {"f1_mittel": float(s.mean()), "f1_sd_seeds": float(s.std(ddof=1)),
                       "f1_median_modell": float(np.median(fz)),
                       "f1_orakel_median": float(np.median([z["f1_orakel"] for z in zz])),
                       "det_je_ref_median": float(np.median([z["det_je_ref"] for z in zz])),
                       "f1_min_modell": float(min(z["f1"] for z in zz)),
                       "f1_max_modell": float(max(z["f1"] for z in zz)),
                       "precision": float(np.mean([z["precision"] for z in zz])),
                       "recall": float(np.mean([z["recall"] for z in zz])),
                       "det_je_ref": float(np.mean([z["det_je_ref"] for z in zz])),
                       "zaehl_r2": float(np.mean([z["zaehl_r2"] for z in zz]))}
            if name in ("21mm", "27mm"):
                spalte = "f1_r12" if name == "21mm" else "f1_r16"
                ind = np.array([float(l[spalte]) for l in laeufe if l["stufe"] == stufe])
                eintrag.update({"ablass_f1_mittel": float(ind.mean()),
                                "ablass_f1_sd_seeds": float(ind.std(ddof=1))})
            zusammen[f"{stufe}_{name}"] = eintrag
            ab = (f" | Ablass {eintrag['ablass_f1_mittel']:.3f} ± {eintrag['ablass_f1_sd_seeds']:.3f}"
                  if "ablass_f1_mittel" in eintrag else "")
            print(f"  {stufe} {name:>6}: Median F1 {eintrag['f1_median_modell']:.3f}, Orakel "
                  f"{eintrag['f1_orakel_median']:.3f}, Det/Ref-Median {eintrag['det_je_ref_median']:.2f}; "
                  f"Mittel F1 {eintrag['f1_mittel']:.3f} ± "
                  f"{eintrag['f1_sd_seeds']:.3f} (P {eintrag['precision']:.3f}, R {eintrag['recall']:.3f}, "
                  f"Det/Ref {eintrag['det_je_ref']:.2f}, Zaehl-r2 {eintrag['zaehl_r2']:.2f}){ab}")
    zusammen["raster"] = {"groesse_px": groesse, "faktor": faktor, "kacheln": len(kacheln),
                          "punkte": int(sum(len(v) for v in ref.values()))}
    (ERG / f"uebertragbarkeit_kv{endung}.json").write_text(json.dumps(zusammen, indent=1), encoding="utf8")
    print(f"\nAusgabe: uebertragbarkeit_kv.csv, uebertragbarkeit_kv.json")


if __name__ == "__main__":
    main()
