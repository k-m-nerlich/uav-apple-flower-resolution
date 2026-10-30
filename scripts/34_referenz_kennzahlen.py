#!/usr/bin/env python3
"""Schritt 34: Referenz-Kennzahlen fuer Version 2 (Review 4.1, 4.3, 4.4).

Alle Vergleiche bei derselben Toleranz wie die Modelle (12 px = 21 mm),
16 px zusaetzlich. Je Vergleich: Precision, Recall, F1 (Referenz = erstes
Argument), Punktzahlen, und fuer die Zaehlung je Kachel Pearson r, ICC(2,1)
(absolute Uebereinstimmung, zweifaktoriell zufaellig) und Bland-Altman
(mittlere Differenz, 95-%-Uebereinstimmungsgrenzen).

Vergleiche
  wiederholung_verblindet   A Runde 1 (nativ) gegen A in der Mehrstufen-
                            Annotation, je Stufe, 12 Kacheln (wie 06)
  wiederholung_53_tage      A Runde 1 gegen A Runde 3, 36 Kacheln
  zweite_person             A Runde 1 gegen B (sicher; zusaetzlich mit unsicher)
  zweite_person_runde3      A Runde 3 gegen B
  modell_*                  DINOv2-Kopf und YOLO + Punkt-NMS (bereinigte KV,
                            native Stufe, 10 Seeds gepoolt) gegen A und B

Mit --ignoriere-unsicher (Version 2) gelten die 'unsicher'-Punkte der jeweiligen
Referenz als Ignorierzonen (wie 21): Vergleichspunkte bzw. Detektionen innerhalb
12 px davon zaehlen weder als Treffer noch als Fehlalarm. Modelle dann mit den
Schwellen aus *_ign_kacheln.csv.

Ausgabe: ergebnisse/referenz_kennzahlen.json bzw. referenz_kennzahlen_ign.json

    python 34_referenz_kennzahlen.py [--ignoriere-unsicher]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gemeinsam import ignorierzone, punkte_aus_export, unsichere_punkte

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
ANN = PROJEKT / "annotation"
RADIEN = (12.0, 16.0)


def modul(datei):
    spec = importlib.util.spec_from_file_location(datei[:2], Path(__file__).with_name(datei))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


KV = modul("21_kv_auswertung.py")
px = lambda l: np.array(l, float).reshape(-1, 2) * 10.24


def naechster(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Abstand jedes Punkts in a zum naechsten Punkt in b (px)."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    return np.sqrt(((a[:, None, :] - b[None, :, :2]) ** 2).sum(-1)).min(1)


def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1) nach Shrout & Fleiss fuer zwei Bewerter."""
    x = np.c_[a, b].astype(float); n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def zaehlung(na: np.ndarray, nb: np.ndarray) -> dict:
    d = nb - na
    return {"r": float(np.corrcoef(na, nb)[0, 1]), "icc_2_1": icc_2_1(na, nb),
            "ba_mittel": float(d.mean()), "ba_grenze_unten": float(d.mean() - 1.96 * d.std(ddof=1)),
            "ba_grenze_oben": float(d.mean() + 1.96 * d.std(ddof=1)),
            "summe_ref": int(na.sum()), "summe_vergleich": int(nb.sum())}


def gefiltert(det: np.ndarray, ref_px: np.ndarray, unsicher: dict | None, k: str) -> np.ndarray:
    if unsicher is None:
        return det
    return det[ignorierzone(det, ref_px, px(unsicher.get(k, [])), 12.0)]


def vergleich(ref: dict, neu: dict, kacheln: list[str], unsicher: dict | None = None) -> dict:
    aus = {}
    neu = {k: gefiltert(px(neu[k]), px(ref[k]), unsicher, k) for k in kacheln}
    for r in RADIEN:
        s = np.zeros(3)
        for k in kacheln:
            s += KV.zuordnen(px(ref[k]), neu[k], r)
        tp, fp, fn = s
        aus[f"r{int(r)}"] = {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": KV.f1(tp, fp, fn)}
    aus["zaehlung"] = zaehlung(np.array([len(ref[k]) for k in kacheln]),
                               np.array([len(neu[k]) for k in kacheln]))
    return aus


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ignoriere-unsicher", action="store_true")
    ign = p.parse_args().ignoriere_unsicher
    lade = lambda d, aus=("unsicher",): {k: v["punkte"] for k, v in punkte_aus_export(ANN / d, ausschluss=aus).items()}
    a1, a3 = lade("export.json"), lade("export_runde3.json")
    b, b_alle = lade("export_annotator2.json"), lade("export_annotator2.json", ())
    u1 = u3 = ub = None
    if ign:
        u1, u3, ub = (unsichere_punkte(ANN / d) for d in ("export.json", "export_runde3.json", "export_annotator2.json"))
    kacheln = sorted(set(a1) & set(a3) & set(b))
    erg = {"kacheln": len(kacheln), "ignorierzonen": ign,
           "wiederholung_53_tage": vergleich(a1, a3, kacheln, u1),
           "zweite_person": vergleich(a1, b, kacheln, u1),
           "zweite_person_mit_unsicher": vergleich(a1, b_alle, kacheln, u1),
           "zweite_person_runde3": vergleich(a3, b, kacheln, u3),
           "zweite_person_als_referenz": vergleich(b, a1, kacheln, ub)}

    # verblindete Wiederholung je Stufe (Mehrstufen-Export, Schluessel wie 06)
    schl = list(csv.DictReader(open(PROJEKT / "kacheln_mehrstufig" / "schluessel.csv", encoding="utf8")))
    ms = lade("export_mehrstufig.json")
    je = defaultdict(lambda: {"ref": {}, "neu": {}})
    for z in schl:
        if z["anonym"] in ms:
            je[float(z["hoehe_m"])]["ref"][z["kachel"]] = a1[z["kachel"]]
            je[float(z["hoehe_m"])]["neu"][z["kachel"]] = ms[z["anonym"]]
    erg["wiederholung_verblindet"] = {f"{h:.1f}": vergleich(d["ref"], d["neu"], sorted(d["ref"]), u1)
                                      for h, d in sorted(je.items())}

    # Modelle (native Stufe, Schwelle je Lauf auf val bei 12 px), gegen A und B
    sx = "_ign" if ign else ""
    for name, kv, kd, nms in (("modell_dinov2", "kv_ssl_dinov2_s", f"kv_ssl_dinov2_s{sx}_kacheln.csv", 0),
                              ("modell_yolo_nms", "kv_bereinigt", f"kv_bereinigt_pnms12{sx}_kacheln.csv", 12),
                              ("modell_dinov2_x15", "kv_ssl_dinov2_s_x1.5", f"kv_ssl_dinov2_s_x1.5{sx}_kacheln.csv", 0)):
        if not (ERG / kd).exists():
            continue
        conf = {(int(z["seed"]), z["kachel"]): float(z["conf"]) for z in csv.DictReader(open(ERG / kd, encoding="utf8"))
                if z["stufe"] == "h051" and float(z["radius_px"]) == 12.0}
        s = {r: {"A": np.zeros(3), "B": np.zeros(3)} for r in RADIEN}
        gef = np.zeros(4)   # Konsens gefunden, Konsens gesamt, nur A gefunden, nur A gesamt
        for f in sorted((ERG / kv / "vorhersagen").glob("h051_*.json")):
            d = json.loads(f.read_text(encoding="utf8"))
            for k, v in d["test"].items():
                det = np.array(v, float).reshape(-1, 3); det = det[det[:, 2] >= 0.05]
                if nms:
                    det = KV.punkt_nms(det, nms)
                det = det[det[:, 2] >= conf[(d["seed"], k)], :2]
                da, db = gefiltert(det, px(a1[k]), u1, k), gefiltert(det, px(b[k]), ub, k)
                # Bluete von A gilt als Konsens, wenn B innerhalb 12 px einen Punkt hat;
                # gefunden, wenn eine Detektion innerhalb 12 px liegt
                pa = px(a1[k])
                kons = naechster(pa, px(b[k])) <= 12
                treff = naechster(pa, det) <= 12
                gef += [(kons & treff).sum(), kons.sum(), (~kons & treff).sum(), (~kons).sum()]
                for r in RADIEN:
                    s[r]["A"] += KV.zuordnen(px(a1[k]), da, r)
                    s[r]["B"] += KV.zuordnen(px(b[k]), db, r)
        erg[name] = {f"r{int(r)}": {ref: {"precision": t[0] / (t[0] + t[1]), "recall": t[0] / (t[0] + t[2]),
                                          "f1": KV.f1(*t)} for ref, t in s[r].items()} for r in RADIEN}
        erg[name]["recall_konsens_r12"] = float(gef[0] / gef[1])
        erg[name]["recall_nur_a_r12"] = float(gef[2] / gef[3])

    (ERG / f"referenz_kennzahlen{sx}.json").write_text(json.dumps(erg, indent=1), encoding="utf8")
    f = lambda x: f"{x:.3f}"
    for k in ("wiederholung_53_tage", "zweite_person", "zweite_person_mit_unsicher", "zweite_person_runde3",
              "zweite_person_als_referenz"):
        e = erg[k]
        print(f"{k:<28} r12: P {f(e['r12']['precision'])} R {f(e['r12']['recall'])} F1 {f(e['r12']['f1'])} | "
              f"r16 F1 {f(e['r16']['f1'])} | Zaehlung r {e['zaehlung']['r']:.2f} ICC {e['zaehlung']['icc_2_1']:.2f} "
              f"BA {e['zaehlung']['ba_mittel']:+.1f} [{e['zaehlung']['ba_grenze_unten']:+.1f}; {e['zaehlung']['ba_grenze_oben']:+.1f}]")
    for h, e in erg["wiederholung_verblindet"].items():
        print(f"verblindet {h:>5} m: r12 F1 {f(e['r12']['f1'])} (P {f(e['r12']['precision'])}, R {f(e['r12']['recall'])}) | r16 F1 {f(e['r16']['f1'])} | Punkte {e['zaehlung']['summe_vergleich']}/{e['zaehlung']['summe_ref']}")
    for k in [k for k in ("modell_dinov2", "modell_yolo_nms", "modell_dinov2_x15") if k in erg]:
        for r in ("r12", "r16"):
            e = erg[k][r]
            print(f"{k:<16} {r}: gegen A F1 {f(e['A']['f1'])} (P {f(e['A']['precision'])}, R {f(e['A']['recall'])}) | "
                  f"gegen B F1 {f(e['B']['f1'])} (P {f(e['B']['precision'])}, R {f(e['B']['recall'])})")
        print(f"{k:<16} Recall Konsens {erg[k]['recall_konsens_r12']:.3f}, nur A {erg[k]['recall_nur_a_r12']:.3f}")


if __name__ == "__main__":
    main()
