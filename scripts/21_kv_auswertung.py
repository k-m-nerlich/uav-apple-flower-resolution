#!/usr/bin/env python3
"""Schritt 21: Kreuzvalidierung auswerten - ohne GPU, nur aus Rohvorhersagen.

Liest ergebnisse/kv/vorhersagen/*.json aus 20_kreuzvalidierung.py.

Betriebspunkt
-------------
Die Konfidenzschwelle wird je Lauf (Stufe x Falte x Seed) auf der
VALIDIERUNGSfalte gewaehlt (bestes F1) und unveraendert auf die Testfalte
angewendet. Zum Vergleich wird auch die alte Praxis gerechnet (Schwelle auf
den Testkacheln selbst gewaehlt, "Orakel"), um den Optimismus zu beziffern.

Kennzahlen je Stufe und Seed, jeweils ueber alle 36 Testkacheln gepoolt
------------------------------------------------------------------------
  F1 bei r = 12 px      Primaerkennzahl, wie 07/08
  F1 bei r = 16 px      Radius der menschlichen Decke (06), fuer den Vergleich
  AP bei r = 12 px      schwellenfrei, Detektionen ab conf 0,05,
                        Zuordnung in Konfidenzreihenfolge (COCO-artig)
  Zaehlfehler           je Kachel Detektionen am Betriebspunkt gegen
                        Referenzpunkte: MAE, Bias, R^2 (Identitaetslinie) und
                        r^2 (Regression, wie Tubau Comas 2019)

Statistik
---------
Gepaarter Cluster-Bootstrap (B = 2000): gezogen werden Kachelgruppen (gleiches
Quellbild oder Bodenueberlappung, ergebnisse/kv/kachelgruppen.json) mit
Zuruecklegen, und zwar fuer alle Stufen DIESELBEN; Seeds werden je Stufe mit
Zuruecklegen gezogen. Ohne Gruppendatei wird je Kachel gezogen. Daraus:
  * Steigung F1 ueber gemessener GSD, 95-%-Intervall
  * Differenz groebste minus feinste Stufe, 90-%-Intervall
  * Aequivalenz (TOST, alpha 0,05): liegt das 90-%-Intervall der Differenz
    und der ueber den ganzen Bereich extrapolierten Steigung vollstaendig in
    +-MARGE, ist ein Aufloesungseffekt dieser Groesse ausgeschlossen.

Die Aequivalenzmarge ist am 15.10.2026 VOR Ansicht der Ergebnisse auf
0,05 F1 gesetzt worden (Groessenordnung der mittleren Lauf-zu-Lauf-Spanne aus
streuung_erweitert.csv, 0,064). Im Paper so berichten; nachtraeglich nicht
anpassen.

Ausgabe, Praefix = Name des KV-Ordners (--kv, z. B. kv oder kv_bereinigt):
    ergebnisse/<kv>_kacheln.csv     je Lauf und Kachel
    ergebnisse/<kv>_laeufe.csv      je Stufe und Seed
    ergebnisse/<kv>_stufen.csv      je Stufe
    ergebnisse/<kv>_statistik.json  Bootstrap und Aequivalenz
    ergebnisse/<kv>_aufloesung.png

kv           erste KV, Falten ohne Ruecksicht auf Bodenueberlappung
kv_bereinigt purged KV: je Falte sind Kacheln, die sich am Boden mit einer
             Testkachel ueberlappen, aus train/val entfernt (massgeblich)

    python 21_kv_auswertung.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gemeinsam import ignorierzone, punkte_aus_export, unsichere_punkte

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
# bis 0,99: mit 0,70 als Obergrenze lag die auf val gewaehlte Schwelle in der
# bereinigten KV mehrfach am Rand (breit feuernde Modelle, 4 val-Kacheln)
CONF_GITTER = np.round(np.concatenate([np.arange(0.05, 0.9501, 0.025), [0.97, 0.99]]), 3)
CONF_MIN = 0.05
RADIEN = (12.0, 16.0)


def zuordnen(wahrheit: np.ndarray, vorhersage: np.ndarray, radius: float):
    """Greedy nach Distanz, identisch zu 07/08. Rueckgabe (TP, FP, FN)."""
    if len(wahrheit) == 0:
        return 0, len(vorhersage), 0
    if len(vorhersage) == 0:
        return 0, 0, len(wahrheit)
    d = np.linalg.norm(wahrheit[:, None, :] - vorhersage[None, :, :], axis=2)
    frei_w = np.ones(len(wahrheit), bool)
    frei_v = np.ones(len(vorhersage), bool)
    tp = 0
    for iw, iv in sorted(np.argwhere(d <= radius).tolist(),
                         key=lambda p: d[p[0], p[1]]):
        if frei_w[iw] and frei_v[iv]:
            frei_w[iw] = frei_v[iv] = False
            tp += 1
    return tp, int(frei_v.sum()), int(frei_w.sum())


def punkt_nms(det: np.ndarray, radius: float) -> np.ndarray:
    """Greedy nach Konfidenz: jede Detektion unterdrueckt schwaechere im Radius."""
    if len(det) < 2:
        return det
    det = det[np.argsort(-det[:, 2])]
    behalten = np.ones(len(det), bool)
    for i in range(len(det)):
        if behalten[i]:
            nah = np.hypot(det[i + 1:, 0] - det[i, 0], det[i + 1:, 1] - det[i, 1]) <= radius
            behalten[i + 1:][nah] = False
    return det[behalten]


def f1(tp, fp, fn) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def ap_treffer(wahrheit: np.ndarray, det: np.ndarray, radius: float):
    """COCO-artig: Detektionen absteigend nach conf, jede an den naechsten
    freien Referenzpunkt. Rueckgabe (conf, ist_tp) je Detektion."""
    if len(det) == 0:
        return np.zeros(0), np.zeros(0, bool)
    det = det[np.argsort(-det[:, 2])]
    frei = np.ones(len(wahrheit), bool)
    treffer = np.zeros(len(det), bool)
    for i, (x, y, _) in enumerate(det):
        if not frei.any():
            break
        d = np.hypot(wahrheit[:, 0] - x, wahrheit[:, 1] - y)
        d[~frei] = np.inf
        j = int(np.argmin(d))
        if d[j] <= radius:
            frei[j] = False
            treffer[i] = True
    return det[:, 2], treffer


def average_precision(conf: np.ndarray, treffer: np.ndarray, n_ref: int) -> float:
    if n_ref == 0 or len(conf) == 0:
        return 0.0
    o = np.argsort(-conf)
    tp = np.cumsum(treffer[o]); fp = np.cumsum(~treffer[o])
    rec = tp / n_ref; prec = tp / (tp + fp)
    prec = np.maximum.accumulate(prec[::-1])[::-1]       # Huellkurve
    rec = np.concatenate([[0.0], rec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec))


def zaehlmasse(det: np.ndarray, ref: np.ndarray) -> dict:
    fehler = det - ref
    ss_tot = float(((ref - ref.mean()) ** 2).sum())
    r2_id = 1 - float((fehler ** 2).sum()) / ss_tot if ss_tot else float("nan")
    r = np.corrcoef(det, ref)[0, 1] if det.std() and ref.std() else float("nan")
    return {"mae": float(np.abs(fehler).mean()),
            "bias_rel": float(fehler.sum() / ref.sum()),
            "r2_identitaet": r2_id, "r2_regression": float(r ** 2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path,
                   default=PROJEKT / "annotation" / "export.json")
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--marge", type=float, default=0.05)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--zufall", type=int, default=20261015)
    p.add_argument("--muster", default="*.json",
                   help="Auswahl der Vorhersagedateien (zum Testen)")
    p.add_argument("--kv", default="kv_bereinigt",
                   help="KV-Ordner unter ergebnisse/ (kv oder kv_bereinigt)")
    p.add_argument("--gruppen", type=Path,
                   default=ERG / "kv" / "kachelgruppen.json",
                   help="Kachelgruppen fuer den Cluster-Bootstrap")
    p.add_argument("--punkt-nms", type=float, default=0.0,
                   help="Detektionen innerhalb dieses Radius (px) zusammenfassen, "
                        "hoechste Konfidenz bleibt. YOLO setzt mit 24-px-Pseudoboxen "
                        "oft 2-3 Boxen auf eine Bluete, die Box-NMS greift dort nicht. "
                        "0 = aus (Stand bis 20.10.2026)")
    p.add_argument("--suffix", default=None,
                   help="Anhang an die Ausgabedateien; Standard: _pnms<R> bei Punkt-NMS")
    p.add_argument("--ignoriere-unsicher", action="store_true",
                   help="Detektionen innerhalb 12 px eines 'unsicher'-Punkts (und nicht "
                        "innerhalb 12 px eines sicheren) weder als Treffer noch als "
                        "Fehlalarm werten (Review 3.4, Version 2); Suffix _ign")
    args = p.parse_args()
    if args.suffix is None:
        args.suffix = (f"_pnms{args.punkt_nms:g}" if args.punkt_nms > 0 else "") +                       ("_ign" if args.ignoriere_unsicher else "")

    kv = ERG / args.kv
    falten = json.loads((kv / "falten.json").read_text(encoding="utf8"))["falten"]
    kacheln = sorted(x for f in falten for x in f)
    roh = punkte_aus_export(args.export)
    ref = {n: np.array([(x * args.kachel_px / 100, y * args.kachel_px / 100)
                        for x, y in roh[n]["punkte"]]).reshape(-1, 2)
           for n in kacheln}

    laeufe = {}
    for pfad in sorted((kv / "vorhersagen").glob(args.muster)):
        d = json.loads(pfad.read_text(encoding="utf8"))
        laeufe[pfad.stem] = d
    if not laeufe:
        raise SystemExit("Keine Vorhersagen gefunden.")

    # Kappung bei max_det pruefen: Liste muss oberhalb CONF_MIN vollstaendig sein
    for name, d in laeufe.items():
        for teil in ("test", "val"):
            for k, v in d[teil].items():
                if len(v) >= d.get("max_det", 1200) and min(x[2] for x in v) >= CONF_MIN:
                    raise SystemExit(f"{name}/{k}: max_det kappt oberhalb conf "
                                     f"{CONF_MIN} - Auswertung ungueltig")

    unsicher = {}
    if args.ignoriere_unsicher:
        u = unsichere_punkte(args.export)
        unsicher = {n: np.array(u.get(n, []), float).reshape(-1, 2) * args.kachel_px / 100
                    for n in kacheln}

    def det_array(v, kachel=None):
        a = np.array(v, dtype=float).reshape(-1, 3)
        a = a[a[:, 2] >= CONF_MIN]
        a = punkt_nms(a, args.punkt_nms) if args.punkt_nms > 0 else a
        if args.ignoriere_unsicher and kachel is not None:
            a = a[ignorierzone(a[:, :2], ref[kachel], unsicher[kachel], 12.0)]
        return a

    # ---------- je Lauf: Schwelle auf val, Zaehlung auf test ----------
    kachelzeilen = []
    ap_sammel = defaultdict(lambda: ([], [], 0))     # (stufe, seed) -> conf, tp, n_ref
    orakel = defaultdict(lambda: np.zeros((len(CONF_GITTER), 3)))
    for name, d in sorted(laeufe.items()):
        stufe = name.split("_")[0]
        val = {k: det_array(v, k) for k, v in d["val"].items()}
        test = {k: det_array(v, k) for k, v in d["test"].items()}
        for r in RADIEN:
            summe = np.zeros((len(CONF_GITTER), 3))
            for k, a in val.items():
                for i, c in enumerate(CONF_GITTER):
                    summe[i] += zuordnen(ref[k], a[a[:, 2] >= c, :2], r)
            c_wahl = float(CONF_GITTER[int(np.argmax([f1(*s) for s in summe]))])
            for k, a in test.items():
                tp, fp, fn = zuordnen(ref[k], a[a[:, 2] >= c_wahl, :2], r)
                kachelzeilen.append({
                    "stufe": stufe, "hoehe_m": d["hoehe_m"],
                    "gsd_gemessen": d["gsd_gemessen"], "seed": d["seed"],
                    "falte": d["falte"], "radius_px": r, "kachel": k,
                    "conf": c_wahl, "TP": tp, "FP": fp, "FN": fn,
                    "detektionen": int((a[:, 2] >= c_wahl).sum()),
                    "referenz": len(ref[k])})
                if r == RADIEN[0]:
                    for i, c in enumerate(CONF_GITTER):
                        orakel[(stufe, d["seed"], d["falte"])][i] += zuordnen(
                            ref[k], a[a[:, 2] >= c, :2], r)
        for k, a in test.items():
            c, t = ap_treffer(ref[k], a, RADIEN[0])
            alt = ap_sammel[(stufe, d["seed"])]
            ap_sammel[(stufe, d["seed"])] = (alt[0] + [c], alt[1] + [t],
                                             alt[2] + len(ref[k]))

    with open(ERG / f"{args.kv}{args.suffix}_kacheln.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(kachelzeilen[0]))
        w.writeheader(); w.writerows(kachelzeilen)

    # ---------- je Stufe und Seed ----------
    stufen = sorted({z["stufe"] for z in kachelzeilen})
    gsd = {s: next(z["gsd_gemessen"] for z in kachelzeilen if z["stufe"] == s)
           for s in stufen}
    hoehe = {s: next(z["hoehe_m"] for z in kachelzeilen if z["stufe"] == s)
             for s in stufen}
    seeds = sorted({z["seed"] for z in kachelzeilen})
    k_index = {k: i for i, k in enumerate(kacheln)}

    # Zaehlwuerfel: [radius, stufe, seed, kachel, (TP, FP, FN, det, ref)]
    W = np.full((len(RADIEN), len(stufen), len(seeds), len(kacheln), 5), np.nan)
    for z in kachelzeilen:
        W[RADIEN.index(z["radius_px"]), stufen.index(z["stufe"]),
          seeds.index(z["seed"]), k_index[z["kachel"]]] = (
            z["TP"], z["FP"], z["FN"], z["detektionen"], z["referenz"])
    vollstaendig = ~np.isnan(W[0, :, :, :, 0]).any(axis=2)   # [stufe, seed]
    if not vollstaendig.all():
        fehlt = [(stufen[i], seeds[j]) for i, j in zip(*np.where(~vollstaendig))]
        print(f"Hinweis: unvollstaendige Stufe/Seed-Paare werden ausgelassen: {fehlt}")

    laufzeilen = []
    for i, s in enumerate(stufen):
        for j, sd in enumerate(seeds):
            if not vollstaendig[i, j]:
                continue
            t12 = W[0, i, j].sum(axis=0); t16 = W[1, i, j].sum(axis=0)
            c, t, n = ap_sammel[(s, sd)]
            # Orakel je Falte: Schwelle auf der Testfalte selbst gewaehlt
            # (alte Praxis), Zaehlungen dann ueber die Falten gepoolt
            o = sum(orakel[(s, sd, k)][int(np.argmax([f1(*x) for x in orakel[(s, sd, k)]]))]
                    for k in range(len(falten)))
            laufzeilen.append({
                "stufe": s, "hoehe_m": hoehe[s], "gsd_gemessen": gsd[s],
                "seed": sd, "f1_r12": f1(*t12[:3]), "f1_r16": f1(*t16[:3]),
                "precision_r12": t12[0] / (t12[0] + t12[1]),
                "recall_r12": t12[0] / (t12[0] + t12[2]),
                "ap_r12": average_precision(np.concatenate(c), np.concatenate(t), n),
                "f1_r12_orakel": f1(*o),
                **{f"zaehl_{k}": v for k, v in
                   zaehlmasse(W[0, i, j, :, 3], W[0, i, j, :, 4]).items()}})
    with open(ERG / f"{args.kv}{args.suffix}_laeufe.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(laufzeilen[0]))
        w.writeheader()
        w.writerows([{k: round(v, 4) if isinstance(v, float) else v
                      for k, v in z.items()} for z in laufzeilen])

    # ---------- je Stufe ----------
    menschlich = {}
    md = ERG / "menschliche_decke.csv"
    if md.exists():
        for z in csv.DictReader(open(md, encoding="utf8")):
            tp, fp, fn = (int(z[k]) for k in ("wiedergefunden", "zusaetzlich", "verfehlt"))
            menschlich[round(float(z["hoehe_m"]), 1)] = f1(tp, fp, fn)
    mensch_kacheln = []
    sl = PROJEKT / "kacheln_mehrstufig" / "schluessel.csv"
    if sl.exists():
        mensch_kacheln = sorted({z["kachel"] for z in
                                 csv.DictReader(open(sl, encoding="utf8"))})
    m_idx = [k_index[k] for k in mensch_kacheln if k in k_index]

    stufenzeilen = []
    print(f"\n{'Stufe':>6} {'GSD':>5} {'n':>3} {'F1 r12':>14} {'Orakel':>7} "
          f"{'F1 r16':>7} {'AP':>6} {'Zaehl-r2':>8} {'Bias':>7} "
          f"{'r16 12K':>8} {'Mensch':>7}")
    for i, s in enumerate(stufen):
        zz = [z for z in laufzeilen if z["stufe"] == s]
        if not zz:
            continue
        m = lambda k: float(np.mean([z[k] for z in zz]))
        sd = lambda k: float(np.std([z[k] for z in zz], ddof=1)) if len(zz) > 1 else 0.0
        # Faltenstreuung: F1 je Falte (ueber Seeds gemittelt), SD ueber Falten
        je_falte = [np.mean([f1(*W[0, i, j, [k_index[k] for k in falte], :3].sum(0))
                             for j in range(len(seeds)) if vollstaendig[i, j]])
                    for falte in falten]
        # Modell auf denselben 12 Kacheln wie die menschliche Decke, r = 16
        r16_12 = (np.mean([f1(*W[1, i, j, m_idx, :3].sum(0))
                           for j in range(len(seeds)) if vollstaendig[i, j]])
                  if m_idx else float("nan"))
        zeile = {"stufe": s, "hoehe_m": hoehe[s], "gsd_gemessen": gsd[s], "n_seeds": len(zz),
                 "f1_r12_mittel": m("f1_r12"), "f1_r12_sd_seeds": sd("f1_r12"),
                 "f1_r12_sd_falten": float(np.std(je_falte, ddof=1)),
                 "f1_r12_orakel_mittel": m("f1_r12_orakel"),
                 "f1_r16_mittel": m("f1_r16"), "ap_r12_mittel": m("ap_r12"),
                 "zaehl_mae": m("zaehl_mae"), "zaehl_bias_rel": m("zaehl_bias_rel"),
                 "zaehl_r2_identitaet": m("zaehl_r2_identitaet"),
                 "zaehl_r2_regression": m("zaehl_r2_regression"),
                 "f1_r16_12kacheln": float(r16_12),
                 "mensch_f1_r16_12kacheln": menschlich.get(round(hoehe[s], 1), float("nan"))}
        stufenzeilen.append(zeile)
        print(f"{s:>6} {gsd[s]:5.2f} {len(zz):3d} "
              f"{zeile['f1_r12_mittel']:.3f}±{zeile['f1_r12_sd_seeds']:.3f} "
              f"{zeile['f1_r12_orakel_mittel']:7.3f} {zeile['f1_r16_mittel']:7.3f} "
              f"{zeile['ap_r12_mittel']:6.3f} {zeile['zaehl_r2_regression']:8.3f} "
              f"{zeile['zaehl_bias_rel']:+7.3f} {zeile['f1_r16_12kacheln']:8.3f} "
              f"{zeile['mensch_f1_r16_12kacheln']:7.3f}")
    with open(ERG / f"{args.kv}{args.suffix}_stufen.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(stufenzeilen[0]))
        w.writeheader()
        w.writerows([{k: round(v, 4) if isinstance(v, float) else v
                      for k, v in z.items()} for z in stufenzeilen])

    # ---------- gepaarter Bootstrap ----------
    ok_stufen = [i for i in range(len(stufen)) if vollstaendig[i].any()]
    x = np.array([gsd[stufen[i]] for i in ok_stufen])
    T = W[0][ok_stufen][:, :, :, :3]                  # [stufe, seed, kachel, 3]
    seed_ok = [np.where(vollstaendig[i])[0] for i in ok_stufen]
    rng = np.random.default_rng(args.zufall)

    def kennzahlen(kachel_idx, seed_idx):
        mittel = []
        for a, i in enumerate(ok_stufen):
            s = T[a][seed_idx[a]][:, kachel_idx].sum(axis=1)     # [seed, 3]
            mittel.append(np.mean(2 * s[:, 0] / (2 * s[:, 0] + s[:, 1] + s[:, 2])))
        mittel = np.array(mittel)
        steigung = np.polyfit(x, mittel, 1)[0]
        return steigung, mittel[-1] - mittel[0], mittel.max() - mittel.min()

    if args.gruppen and args.gruppen.exists():
        gruppen = [[k_index[k] for k in g] for g in
                   json.loads(args.gruppen.read_text(encoding="utf8"))["gruppen"]]
    else:
        gruppen = [[i] for i in range(len(kacheln))]
    assert sorted(i for g in gruppen for i in g) == list(range(len(kacheln)))

    def ziehe_kacheln():
        gezogen = rng.integers(0, len(gruppen), len(gruppen))
        return np.array([i for g in gezogen for i in gruppen[g]])

    punkt = kennzahlen(np.arange(len(kacheln)), seed_ok)
    boot = np.array([kennzahlen(ziehe_kacheln(),
                                [rng.choice(s_, len(s_)) for s_ in seed_ok])
                     for _ in range(args.bootstrap)])
    bereich = x.max() - x.min()
    q = lambda v, a, b: [float(np.quantile(v, a)), float(np.quantile(v, b))]
    stat = {
        "stufen": [stufen[i] for i in ok_stufen], "gsd_gemessen": x.tolist(),
        "seeds_je_stufe": [len(s_) for s_ in seed_ok], "kacheln": len(kacheln),
        "bootstrap_einheiten": len(gruppen),
        "bootstrap": args.bootstrap, "marge_f1": args.marge,
        "steigung_f1_je_mm": float(punkt[0]),
        "steigung_ki95": q(boot[:, 0], .025, .975),
        "aenderung_ueber_bereich": float(punkt[0] * bereich),
        "aenderung_ueber_bereich_ki90": [v * bereich for v in q(boot[:, 0], .05, .95)],
        "differenz_grob_minus_fein": float(punkt[1]),
        "differenz_ki90": q(boot[:, 1], .05, .95),
        "differenz_ki95": q(boot[:, 1], .025, .975),
        "spanne_stufenmittel": float(punkt[2]),
        "spanne_stufenmittel_ki95": q(boot[:, 2], .025, .975),
    }
    stat["aequivalent_differenz"] = bool(
        -args.marge < stat["differenz_ki90"][0] and stat["differenz_ki90"][1] < args.marge)
    stat["aequivalent_steigung"] = bool(
        -args.marge < stat["aenderung_ueber_bereich_ki90"][0]
        and stat["aenderung_ueber_bereich_ki90"][1] < args.marge)
    (ERG / f"{args.kv}{args.suffix}_statistik.json").write_text(json.dumps(stat, indent=1), encoding="utf8")

    print(f"\nSteigung        : {stat['steigung_f1_je_mm']:+.4f} F1 je mm/px, "
          f"95-%-KI [{stat['steigung_ki95'][0]:+.4f}, {stat['steigung_ki95'][1]:+.4f}]")
    print(f"ueber {bereich:.2f} mm/px : {stat['aenderung_ueber_bereich']:+.4f}, 90-%-KI "
          f"[{stat['aenderung_ueber_bereich_ki90'][0]:+.4f}, "
          f"{stat['aenderung_ueber_bereich_ki90'][1]:+.4f}]  -> aequivalent "
          f"(+-{args.marge}): {stat['aequivalent_steigung']}")
    print(f"grob - fein     : {stat['differenz_grob_minus_fein']:+.4f}, 90-%-KI "
          f"[{stat['differenz_ki90'][0]:+.4f}, {stat['differenz_ki90'][1]:+.4f}]  "
          f"-> aequivalent: {stat['aequivalent_differenz']}")
    print(f"Spanne Mittel   : {stat['spanne_stufenmittel']:.4f}")

    # ---------- Abbildung ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for z in stufenzeilen:
        werte = [l["f1_r12"] for l in laufzeilen if l["stufe"] == z["stufe"]]
        ax.plot([z["gsd_gemessen"]] * len(werte), werte, "o", color="#5d6d7e",
                alpha=.45, markersize=6,
                label="Einzelseeds (KV, 36 Testkacheln)" if z is stufenzeilen[0] else None)
    gx = [z["gsd_gemessen"] for z in stufenzeilen]
    ax.plot(gx, [z["f1_r12_mittel"] for z in stufenzeilen], "-", color="#c0392b",
            linewidth=2.4, label="F1 Mittel (r = 12 px)")
    ax.plot(gx, [z["ap_r12_mittel"] for z in stufenzeilen], "s--", color="#1a5276",
            linewidth=1.6, label="AP (r = 12 px)")
    ax.plot(gx, [z["f1_r16_12kacheln"] for z in stufenzeilen], "d-", color="#b9770e",
            linewidth=1.6, label="F1 Modell, r = 16 px, 12 Kacheln der Decke")
    ax.plot(gx, [z["mensch_f1_r16_12kacheln"] for z in stufenzeilen], "d:",
            color="#196f3d", linewidth=1.8, label="F1 Mensch, r = 16 px, 12 Kacheln")
    ax.set_xlabel("Bodenaufloesung (mm/px, gemessen)")
    ax.set_ylabel("Guete")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Kreuzvalidierung, {len(seeds)} Seeds x 6 Falten je Stufe\n"
                 f"Steigung {stat['steigung_f1_je_mm']:+.3f} F1 je mm/px, 95-%-KI "
                 f"[{stat['steigung_ki95'][0]:+.3f}, {stat['steigung_ki95'][1]:+.3f}]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(ERG / f"{args.kv}{args.suffix}_aufloesung.png", dpi=150)
    print(f"\nAusgabe: {args.kv}{args.suffix}_kacheln.csv, _laeufe.csv, _stufen.csv, "
          f"_statistik.json, _aufloesung.png in {ERG}")


if __name__ == "__main__":
    main()
