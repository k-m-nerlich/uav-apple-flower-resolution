#!/usr/bin/env python3
"""Schritt 23: Farbschwellwert-Baseline in derselben bereinigten Kreuzvalidierung.

Warum neu
---------
Die erste Baseline (Mac, 19_schwellwert_baseline.py, Nummer auf dem PC
anderweitig vergeben) hat den Betriebspunkt auf allen 36 Kacheln gesucht und
auf denselben 36 Kacheln berichtet: F1 0,517, in-sample. Der Detektor wird seit
20/21 kreuzvalidiert berichtet (F1 0,489 bei 1,71 mm/px). So sind die Zahlen
nicht vergleichbar - die Baseline hatte 120 Versuche auf den Testdaten.

Hier bekommt die Baseline dieselbe Behandlung wie das Modell:
  * dieselben Falten und Sperrlisten (ergebnisse/kv_bereinigt/falten.json)
  * Betriebspunkt (hell, sat, Mindestflaeche) je Falte auf train+val gewaehlt,
    unveraendert auf die Testfalte angewendet
  * Zuordnung r = 12 px (Modell) und 16 px (menschliche Decke), ueber alle
    36 Testkacheln gepoolt
  * alle fuenf Stufen, Kacheln degradiert wie in 20 (LANCZOS/BICUBIC)
  * Zaehlmasse je Kachel wie in 21 (r^2 der Regression, Bias)
  * gepaarter Cluster-Bootstrap ueber die 16 Kachelgruppen fuer die Differenz
    Detektor minus Baseline (Detektor: Mittel ueber die 10 Seeds je Kachel)

Das Gitter ist gegenueber 19 nach oben erweitert, weil der beste Punkt dort am
Rand lag (hell 225, sat 25).

Laeuft ohne GPU, rund 3 min. Kacheln aus kacheln/<Flug>/.

Ausgabe
    ergebnisse/schwellwert_kv_stufen.csv   je Stufe und Radius: Baseline,
                                           Detektor, Differenz mit 95-%-KI
    ergebnisse/schwellwert_kv_kacheln.csv  je Stufe, Radius und Kachel
    ergebnisse/schwellwert_kv.json         Betriebspunkte je Falte, in-sample

    python3 23_schwellwert_kv.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gemeinsam import ignorierzone, mm_pro_pixel, punkte_aus_export, unsichere_punkte

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
STUFEN_M = [5.1, 8.8, 10.7, 14.8, 16.7]   # wie 20


def marke_von(hoehe: float) -> str:
    return f"h{round(hoehe * 10):03d}"


def degradieren(bild: Image.Image, faktor: float, kachel_px: int) -> np.ndarray:
    """Wie 20.baue_stufe: LANCZOS herunter, BICUBIC zurueck."""
    if faktor < 0.999:
        klein = max(32, int(round(kachel_px * faktor)))
        bild = (bild.resize((klein, klein), Image.LANCZOS)
                    .resize((kachel_px, kachel_px), Image.BICUBIC))
    return np.asarray(bild)


def zuordnen(wahrheit: np.ndarray, vorhersage: np.ndarray, radius: float):
    """Greedy-Zuordnung nach Distanz, wie 19 und 21. Rueckgabe (TP, FP, FN)."""
    if len(wahrheit) == 0:
        return 0, len(vorhersage), 0
    if len(vorhersage) == 0:
        return 0, 0, len(wahrheit)
    d = np.linalg.norm(wahrheit[:, None, :] - vorhersage[None, :, :], axis=2)
    fw = np.ones(len(wahrheit), bool)
    fv = np.ones(len(vorhersage), bool)
    tp = 0
    for iw, iv in sorted(np.argwhere(d <= radius).tolist(),
                         key=lambda p: d[p[0], p[1]]):
        if fw[iw] and fv[iv]:
            fw[iw] = fv[iv] = False
            tp += 1
    return tp, int(fv.sum()), int(fw.sum())


def f1(tp, fp, fn) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def maske(rgb: np.ndarray, hell_min: int, sat_max: int) -> np.ndarray:
    a = rgb.astype(np.int16)
    hell = a.max(2)
    return ((hell >= hell_min) & ((hell - a.min(2)) <= sat_max)).astype(np.uint8)


def r2_regression(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2) if x.std() > 0 and y.std() > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, default=PROJEKT / "annotation" / "export.json")
    p.add_argument("--kv", default="kv_bereinigt")
    p.add_argument("--gruppen", type=Path, default=ERG / "kv" / "kachelgruppen.json")
    p.add_argument("--stufen", type=float, nargs="+", default=STUFEN_M)
    p.add_argument("--radien", type=float, nargs="+", default=[12.0, 16.0])
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--hell", type=int, nargs="+",
                   default=[180, 195, 210, 225, 235, 245])
    p.add_argument("--sat", type=int, nargs="+", default=[10, 15, 20, 25, 35, 45])
    p.add_argument("--min-flaeche", type=int, nargs="+",
                   default=[10, 20, 35, 50, 75, 100])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--zufall", type=int, default=20261020)
    p.add_argument("--detektor-suffix", default="",
                   help="Detektor-Vergleich aus <kv><suffix>_kacheln.csv (21), z. B. _pnms12_ign")
    p.add_argument("--ignoriere-unsicher", action="store_true",
                   help="Ignorierzonen um 'unsicher'-Punkte wie in 21 (Review 3.4); "
                        "Ausgabe dann schwellwert_kv_ign_*")
    p.add_argument("--ausgabe-suffix", default="",
                   help="haengt an den Ausgabenamen an, z. B. _erw fuer die erweiterten Stufen "
                        "(sonst wuerden die kanonischen Dateien ueberschrieben)")
    args = p.parse_args()
    aus = "schwellwert_kv" + args.ausgabe_suffix + ("_ign" if args.ignoriere_unsicher else "")

    import cv2

    kv = json.loads((ERG / args.kv / "falten.json").read_text(encoding="utf8"))
    falten, teilungen = kv["falten"], kv["teilungen"]
    kacheln = sorted(x for f in falten for x in f)
    k_index = {n: i for i, n in enumerate(kacheln)}
    ordner = [d for d in sorted((PROJEKT / "kacheln").glob("*")) if d.is_dir()][0]
    roh = punkte_aus_export(args.export)
    K = args.kachel_px
    ref = {n: np.array(roh[n]["punkte"], float).reshape(-1, 2) / 100 * K for n in kacheln}
    original = {n: Image.open(ordner / n).convert("RGB") for n in kacheln}
    u = unsichere_punkte(args.export) if args.ignoriere_unsicher else {}
    unsicher = {n: np.array(u.get(n, []), float).reshape(-1, 2) / 100 * K for n in kacheln}
    gruppen = [[k_index[n] for n in g] for g in
               json.loads(args.gruppen.read_text(encoding="utf8"))["gruppen"]]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gitter = [(h, s, m) for h in args.hell for s in args.sat for m in args.min_flaeche]
    print(f"{len(kacheln)} Kacheln, {sum(len(r) for r in ref.values())} Referenzpunkte, "
          f"Gitter {len(gitter)} Betriebspunkte, {len(gruppen)} Kachelgruppen")

    # Detektor je Stufe, Radius und Kachel: [seeds, 3]
    det = {}
    with open(ERG / f"{args.kv}{args.detektor_suffix}_kacheln.csv", encoding="utf8") as f:
        for z in csv.DictReader(f):
            det.setdefault((z["stufe"], float(z["radius_px"]), z["kachel"]), []).append(
                (int(z["TP"]), int(z["FP"]), int(z["FN"])))
    det = {k: np.array(v, float) for k, v in det.items()}

    basis = mm_pro_pixel(STUFEN_M[0])
    rng = np.random.default_rng(args.zufall)
    stufenzeilen, kachelzeilen, wahl_alle = [], [], []
    for hoehe in args.stufen:
        marke = marke_von(hoehe)
        faktor = basis / mm_pro_pixel(hoehe)
        # Zentren und Flaechen je Kachel und (hell, sat), einmal je Stufe
        komp = {}
        for n in kacheln:
            rgb = degradieren(original[n], faktor, K)
            for h in args.hell:
                for s in args.sat:
                    rein = cv2.morphologyEx(maske(rgb, h, s), cv2.MORPH_OPEN, kernel)
                    _, _, st, z = cv2.connectedComponentsWithStats(rein, connectivity=8)
                    komp[(n, h, s)] = (z[1:], st[1:, 4])

        for r in args.radien:
            tab = {}
            for n in kacheln:
                for gi, (h, s, m) in enumerate(gitter):
                    z, fl = komp[(n, h, s)]
                    v = z[fl >= m]
                    if args.ignoriere_unsicher:
                        v = v[ignorierzone(v, ref[n], unsicher[n], 12.0)]
                    tab[(n, gi)] = (*zuordnen(ref[n], v, r), len(v))

            def gepoolt(namen, gi):
                return f1(*np.sum([tab[(n, gi)][:3] for n in namen], axis=0))

            gi_in = int(np.argmax([gepoolt(kacheln, gi) for gi in range(len(gitter))]))
            B = {}
            for k, (test, teil) in enumerate(zip(falten, teilungen)):
                abstimm = teil["train"] + teil["val"]
                f_ab = [gepoolt(abstimm, gi) for gi in range(len(gitter))]
                gi = int(np.argmax(f_ab))
                wahl_alle.append({"stufe": marke, "radius_px": r, "falte": k,
                                  "hell": gitter[gi][0], "sat": gitter[gi][1],
                                  "min_flaeche": gitter[gi][2],
                                  "f1_abstimmung": round(f_ab[gi], 4)})
                for n in test:
                    tp, fp, fn, nd = tab[(n, gi)]
                    B[n] = np.array([tp, fp, fn], float)
                    kachelzeilen.append({"stufe": marke, "radius_px": r, "kachel": n,
                                         "falte": k, "TP": tp, "FP": fp, "FN": fn,
                                         "detektionen": nd, "referenz": len(ref[n])})

            T = np.array([B[n] for n in kacheln])
            F_b = f1(*T.sum(0))
            nd = np.array([z["detektionen"] for z in kachelzeilen
                           if z["stufe"] == marke and z["radius_px"] == r], float)
            nr = np.array([z["referenz"] for z in kachelzeilen
                           if z["stufe"] == marke and z["radius_px"] == r], float)

            D = {n: det[(marke, r, n)] for n in kacheln}
            n_seeds = len(D[kacheln[0]])

            def f1_det(idx, sd):
                s = np.sum([D[kacheln[i]][sd] for i in idx], axis=0)
                return float(np.mean([f1(*x) for x in s]))

            def f1_base(idx):
                return f1(*np.sum([B[kacheln[i]] for i in idx], axis=0))

            F_d = f1_det(range(len(kacheln)), np.arange(n_seeds))
            diffs = []
            for _ in range(args.bootstrap):
                idx = [i for g in rng.integers(0, len(gruppen), len(gruppen))
                       for i in gruppen[g]]
                diffs.append(f1_det(idx, rng.integers(0, n_seeds, n_seeds)) - f1_base(idx))
            ki = np.percentile(diffs, [2.5, 97.5])
            stufenzeilen.append({
                "stufe": marke, "hoehe_m": hoehe, "radius_px": r,
                "baseline_f1": round(F_b, 4),
                "baseline_precision": round(T[:, 0].sum() / T[:, :2].sum(), 4),
                "baseline_recall": round(T[:, 0].sum() / T[:, [0, 2]].sum(), 4),
                "baseline_insample_f1": round(gepoolt(kacheln, gi_in), 4),
                "baseline_zaehl_r2": round(r2_regression(nd, nr), 4),
                "baseline_zaehl_bias_rel": round(float((nd - nr).sum() / nr.sum()), 4),
                "detektor_f1": round(F_d, 4),
                "differenz": round(F_d - F_b, 4),
                "differenz_ki95_unten": round(float(ki[0]), 4),
                "differenz_ki95_oben": round(float(ki[1]), 4)})
            z = stufenzeilen[-1]
            print(f"{marke} r={r:.0f}: Baseline {F_b:.3f} (in-sample "
                  f"{z['baseline_insample_f1']:.3f}, Zaehl-r2 {z['baseline_zaehl_r2']:.3f}, "
                  f"Bias {z['baseline_zaehl_bias_rel']:+.1%})  Detektor {F_d:.3f}  "
                  f"Diff {F_d - F_b:+.3f} [{ki[0]:+.3f}; {ki[1]:+.3f}]", flush=True)

    for name, zeilen in ((f"{aus}_stufen.csv", stufenzeilen),
                         (f"{aus}_kacheln.csv", kachelzeilen)):
        with open(ERG / name, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
            w.writeheader(); w.writerows(zeilen)
    (ERG / f"{aus}.json").write_text(json.dumps({
        "kv": args.kv, "detektor_suffix": args.detektor_suffix,
        "ignoriere_unsicher": args.ignoriere_unsicher, "bootstrap": args.bootstrap, "bootstrap_einheiten": len(gruppen),
        "gitter": {"hell": args.hell, "sat": args.sat, "min_flaeche": args.min_flaeche},
        "betriebspunkte": wahl_alle}, indent=1), encoding="utf8")
    print(f"\n-> {ERG / (aus + '_stufen.csv')}")


if __name__ == "__main__":
    main()
