#!/usr/bin/env python3
"""Schritt 22: Inter-Rater-Vergleich vertieft, und das Modell dagegen gelesen.

18_inter_rater.py liefert die Kennzahlen A gegen B. Hier kommt dazu:

1. B mit und ohne "unsicher"-Punkte
2. Radiusabhaengigkeit (10 / 16 / 24 / 32 px)
3. Zaehleinheit: Hat B je Bueschel einen Punkt gesetzt? Dann laegen um jeden
   B-Punkt deutlich mehr A-Punkte als um einen beliebigen A-Punkt.
   Verglichen wird zusaetzlich der mediane Nachbarabstand beider.
4. Modell gegen Konsens: A-Punkte werden geteilt in
       Konsens   B hat innerhalb r = 16 px ebenfalls einen Punkt (sicher
                 oder unsicher)
       nur A     B hat dort nichts gesetzt
   und der Recall des Modells (bereinigte KV, native Stufe, alle 10 Seeds,
   Schwelle auf val gewaehlt, r = 12 px) je Gruppe gemessen. Findet das Modell
   den Konsens deutlich besser als den Rest, liegt ein Teil seiner Fehlbetraege
   in der Mehrdeutigkeit der Referenz, nicht im Modell.
5. Modell gegen B als Referenz (gleiche Vorhersagen, gleiche Schwellen).

Ausgabe: ergebnisse/inter_rater_vertieft.json, Konsole.

    python 22_inter_rater_vertieft.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gemeinsam import punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
MM_PRO_PX = 1.714          # gemessene native GSD


def paare(a: np.ndarray, b: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Greedy nach Distanz wie in 07/18/21, gibt die Indexpaare zurueck."""
    if len(a) == 0 or len(b) == 0:
        return []
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    fa, fb = np.ones(len(a), bool), np.ones(len(b), bool)
    aus = []
    for i, j in sorted(np.argwhere(d <= radius).tolist(), key=lambda p: d[p[0], p[1]]):
        if fa[i] and fb[j]:
            fa[i] = fb[j] = False
            aus.append((i, j))
    return aus


def nachbarabstand(p: np.ndarray) -> np.ndarray:
    if len(p) < 2:
        return np.zeros(0)
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, default=PROJEKT / "annotation" / "export.json")
    p.add_argument("--b", type=Path,
                   default=PROJEKT / "annotation" / "export_annotator2.json")
    p.add_argument("--kv", default="kv_bereinigt")
    p.add_argument("--stufe", default="h051")
    p.add_argument("--kachel-px", type=int, default=1024)
    args = p.parse_args()

    K = args.kachel_px
    px = lambda l: np.array(l, float).reshape(-1, 2) / 100.0 * K
    A = punkte_aus_export(args.a)
    B_sicher = punkte_aus_export(args.b)
    B_alle = punkte_aus_export(args.b, ausschluss=())
    kacheln = sorted(set(A) & set(B_sicher))
    a = {k: px(A[k]["punkte"]) for k in kacheln}
    bs = {k: px(B_sicher[k]["punkte"]) for k in kacheln}
    ba = {k: px(B_alle[k]["punkte"]) for k in kacheln}
    erg = {"kacheln": len(kacheln)}

    # 1 + 2: Uebereinstimmung je Variante und Radius
    print(f"{'B':<14} {'r px':>5} {'r mm':>5} {'A->B':>6} {'B->A':>6} {'F1':>6}")
    tab = []
    for name, b in (("B sicher", bs), ("B mit unsicher", ba)):
        for r in (10, 16, 24, 32):
            tp = sum(len(paare(a[k], b[k], r)) for k in kacheln)
            na = sum(len(a[k]) for k in kacheln); nb = sum(len(b[k]) for k in kacheln)
            ab, ba_ = tp / na, tp / nb
            f1 = 2 * tp / (na + nb)
            tab.append({"b": name, "radius_px": r, "a_nach_b": ab, "b_nach_a": ba_,
                        "f1": f1, "n_a": na, "n_b": nb})
            print(f"{name:<14} {r:5d} {r*MM_PRO_PX:5.0f} {ab:6.3f} {ba_:6.3f} {f1:6.3f}")
    erg["uebereinstimmung"] = tab

    za = np.array([len(a[k]) for k in kacheln]); zb = np.array([len(ba[k]) for k in kacheln])
    zs = np.array([len(bs[k]) for k in kacheln])
    erg["zaehlung"] = {
        "r_a_bsicher": float(np.corrcoef(za, zs)[0, 1]),
        "r_a_balle": float(np.corrcoef(za, zb)[0, 1]),
        "verhaeltnis_bsicher_median": float(np.median(zs[za > 0] / za[za > 0])),
        "verhaeltnis_balle_median": float(np.median(zb[za > 0] / za[za > 0])),
    }
    print(f"\nZaehlkorrelation r: A~B sicher {erg['zaehlung']['r_a_bsicher']:.3f}, "
          f"A~B alle {erg['zaehlung']['r_a_balle']:.3f}; B/A je Kachel (Median) "
          f"{erg['zaehlung']['verhaeltnis_bsicher_median']:.2f} bzw. "
          f"{erg['zaehlung']['verhaeltnis_balle_median']:.2f}")

    # 3: Zaehleinheit
    um_b, um_a = [], []
    for k in kacheln:
        if len(a[k]) == 0:
            continue
        for q in ba[k]:
            um_b.append(int((np.linalg.norm(a[k] - q, axis=1) <= 40).sum()))
        for q in a[k]:
            um_a.append(int((np.linalg.norm(a[k] - q, axis=1) <= 40).sum()))
    nn_a = np.concatenate([nachbarabstand(a[k]) for k in kacheln])
    nn_b = np.concatenate([nachbarabstand(ba[k]) for k in kacheln])
    erg["zaehleinheit"] = {
        "a_punkte_im_umkreis_40px_um_b": float(np.mean(um_b)),
        "a_punkte_im_umkreis_40px_um_a": float(np.mean(um_a)),
        "nachbarabstand_median_mm_a": float(np.median(nn_a) * MM_PRO_PX),
        "nachbarabstand_median_mm_b": float(np.median(nn_b) * MM_PRO_PX),
    }
    print(f"A-Punkte im Umkreis 40 px: um B-Punkte {np.mean(um_b):.2f}, "
          f"um A-Punkte {np.mean(um_a):.2f} (inkl. sich selbst)")
    print(f"Medianer Nachbarabstand: A {np.median(nn_a)*MM_PRO_PX:.0f} mm, "
          f"B {np.median(nn_b)*MM_PRO_PX:.0f} mm")

    # 4 + 5: Modell gegen Konsens und gegen B
    konsens = {}
    for k in kacheln:
        m = np.zeros(len(a[k]), bool)
        for i, _ in paare(a[k], ba[k], 16):
            m[i] = True
        konsens[k] = m
    n_kons = sum(m.sum() for m in konsens.values()); n_a = sum(len(m) for m in konsens.values())
    print(f"\nKonsens (A-Punkte mit B-Partner, r = 16 px): {n_kons} von {n_a} "
          f"({n_kons/n_a:.1%})")

    schwelle = {}
    for z in csv.DictReader(open(ERG / f"{args.kv}_kacheln.csv", encoding="utf8")):
        if z["stufe"] == args.stufe and float(z["radius_px"]) == 12.0:
            schwelle[(int(z["seed"]), z["kachel"])] = float(z["conf"])

    je_seed = defaultdict(lambda: np.zeros(6))   # gefunden/gesamt Konsens, nur A; TP/FP/FN gegen B
    for pfad in sorted((ERG / args.kv / "vorhersagen").glob(f"{args.stufe}_*.json")):
        d = json.loads(pfad.read_text(encoding="utf8"))
        for k, v in d["test"].items():
            det = np.array(v, float).reshape(-1, 3)
            det = det[det[:, 2] >= schwelle[(d["seed"], k)], :2]
            gef = np.zeros(len(a[k]), bool)
            for i, _ in paare(a[k], det, 12):
                gef[i] = True
            m = konsens[k]
            tp_b = len(paare(bs[k], det, 12))
            je_seed[d["seed"]] += [gef[m].sum(), m.sum(), gef[~m].sum(), (~m).sum(),
                                   tp_b, len(det)]
    rk, rn, pb, rb, f1b = [], [], [], [], []
    nb_sicher = sum(len(bs[k]) for k in kacheln)
    for s, v in sorted(je_seed.items()):
        rk.append(v[0] / v[1]); rn.append(v[2] / v[3])
        tp, ndet = v[4], v[5]
        pb.append(tp / ndet); rb.append(tp / nb_sicher)
        f1b.append(2 * tp / (ndet + nb_sicher))
    erg["modell"] = {
        "stufe": args.stufe, "seeds": len(je_seed),
        "recall_konsens": [float(np.mean(rk)), float(np.std(rk, ddof=1))],
        "recall_nur_a": [float(np.mean(rn)), float(np.std(rn, ddof=1))],
        "gegen_b_sicher": {"precision": float(np.mean(pb)), "recall": float(np.mean(rb)),
                           "f1": float(np.mean(f1b))},
    }
    print(f"Modell-Recall ({args.stufe}, {len(je_seed)} Seeds, r = 12 px): "
          f"Konsens {np.mean(rk):.3f} ± {np.std(rk, ddof=1):.3f}, "
          f"nur A {np.mean(rn):.3f} ± {np.std(rn, ddof=1):.3f}")
    print(f"Modell gegen B (sicher) als Referenz: P {np.mean(pb):.3f}, "
          f"R {np.mean(rb):.3f}, F1 {np.mean(f1b):.3f}")

    (ERG / "inter_rater_vertieft.json").write_text(json.dumps(erg, indent=1),
                                                   encoding="utf8")
    print(f"\nAusgabe: {ERG / 'inter_rater_vertieft.json'}")


if __name__ == "__main__":
    main()
