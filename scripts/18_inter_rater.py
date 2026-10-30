#!/usr/bin/env python3
"""Schritt 18: Uebereinstimmung zweier Annotierender auf denselben Kacheln.

Warum das zaehlt
----------------
06_menschliche_decke.py misst, wie gut eine Person ihre eigene Annotation
wiederholt (intra-rater, hier 0.741). Das ist eine Aussage ueber DIESE Person.
Erst ein zweiter Annotator macht daraus eine Aussage ueber die AUFGABE - und
genau darauf zielt der Beitrag.

Berichtet werden
----------------
Wiederfindung A->B   Anteil der Punkte aus A, die B ebenfalls gesetzt hat
Wiederfindung B->A   umgekehrt
F1                   symmetrisches Mass der Uebereinstimmung
Zaehlkorrelation     Zusammenhang der Punktzahlen je Kachel. Fuer die
                     Anwendung oft wichtiger als die Punkt-fuer-Punkt-
                     Uebereinstimmung: Zur Ausduennungsentscheidung zaehlt die
                     relative Bluehstaerke, nicht die exakte Einzelblueten-
                     zuordnung. Beide Zahlen gehoeren ins Paper.
Unsicherheitsquoten  je Annotator

Es gibt kein "richtig" und kein "falsch". A ist nicht die Wahrheit, sondern
nur die zuerst erhobene Runde; deshalb werden beide Richtungen berichtet.

    python3 18_inter_rater.py --a annotation/export.json \
                             --b annotation/export_annotator2.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gemeinsam import punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent


def zuordnen(a: np.ndarray, b: np.ndarray, radius: float) -> int:
    """Groesse der groessten Zuordnung zwischen zwei Punktmengen (greedy)."""
    if len(a) == 0 or len(b) == 0:
        return 0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    frei_a, frei_b = np.ones(len(a), bool), np.ones(len(b), bool)
    tp = 0
    for ia, ib in sorted(np.argwhere(d <= radius).tolist(),
                         key=lambda p: d[p[0], p[1]]):
        if frei_a[ia] and frei_b[ib]:
            frei_a[ia] = frei_b[ib] = False
            tp += 1
    return tp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, default=PROJEKT / "annotation" / "export.json",
                   help="Annotation des ersten Durchgangs")
    p.add_argument("--b", type=Path, required=True,
                   help="Annotation der zweiten Person")
    p.add_argument("--name-a", default="Annotator A")
    p.add_argument("--name-b", default="Annotator B")
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--radius-px", type=float, default=16.0)
    args = p.parse_args()

    for pfad, was in ((args.a, args.name_a), (args.b, args.name_b)):
        if not pfad.exists():
            raise SystemExit(f"Export fehlt ({was}): {pfad}")

    A, B = punkte_aus_export(args.a), punkte_aus_export(args.b)
    gemeinsam = sorted(set(A) & set(B))
    if not gemeinsam:
        raise SystemExit("Keine gemeinsamen Kacheln. Wurden dieselben Dateien "
                         "annotiert? Label Studio haengt beim Upload ein "
                         "Hex-Praefix an, das gemeinsam.ls_dateiname entfernt.")
    nur_a, nur_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if nur_a or nur_b:
        print(f"Hinweis   : {len(nur_a)} Kacheln nur in A, {len(nur_b)} nur in B "
              f"- ausgewertet werden die {len(gemeinsam)} gemeinsamen\n")

    K = args.kachel_px
    proz = lambda l: (np.array(l) / 100.0 * K if l else np.zeros((0, 2)))

    tp = na = nb = 0
    za, zb, zeilen = [], [], []
    for k in gemeinsam:
        pa, pb = proz(A[k]["punkte"]), proz(B[k]["punkte"])
        t = zuordnen(pa, pb, args.radius_px)
        tp += t; na += len(pa); nb += len(pb)
        za.append(len(pa)); zb.append(len(pb))
        zeilen.append({"kachel": k, "n_a": len(pa), "n_b": len(pb),
                       "uebereinstimmend": t,
                       "unsicher_a": A[k]["unsicher"], "unsicher_b": B[k]["unsicher"]})

    r_ab = tp / na if na else 0.0
    r_ba = tp / nb if nb else 0.0
    f1 = 2 * r_ab * r_ba / (r_ab + r_ba) if r_ab + r_ba else 0.0
    za, zb = np.array(za), np.array(zb)
    korr = float(np.corrcoef(za, zb)[0, 1]) if len(za) > 1 else float("nan")
    ua = sum(A[k]["unsicher"] for k in gemeinsam)
    ub = sum(B[k]["unsicher"] for k in gemeinsam)

    print(f"Kacheln            {len(gemeinsam)}")
    breite = max(len(args.name_a), len(args.name_b)) + 2
    print(f"Punkte {args.name_a:<{breite}}{na}")
    print(f"Punkte {args.name_b:<{breite}}{nb}")
    print(f"Zuordnungsradius   {args.radius_px:.0f} px\n")
    print(f"Wiederfindung A->B {r_ab:.3f}")
    print(f"Wiederfindung B->A {r_ba:.3f}")
    print(f"F1 (symmetrisch)   {f1:.3f}")
    print(f"Zaehlkorrelation r {korr:.3f}   (Punktzahlen je Kachel)")
    richtung = "gleich viele" if nb == na else ("mehr" if nb > na else "weniger")
    print(f"Mengenverhaeltnis  {nb/na if na else float('nan'):.3f}   "
          f"(B setzt {richtung} Punkte)")
    print(f"unsicher           {args.name_a}: {100*ua/(na+ua) if na+ua else 0:.1f} %   "
          f"{args.name_b}: {100*ub/(nb+ub) if nb+ub else 0:.1f} %")
    print()
    print("Einordnung: Die Intra-Rater-Uebereinstimmung derselben Person liegt")
    print("bei 0.741 (06_menschliche_decke.py). Liegt der Wert hier deutlich")
    print("darunter, ist die Aufgabe selbst mehrdeutig und nicht nur diese")
    print("Person inkonsistent - das ist die staerkere Aussage fuer das Paper.")
    print("Liegt er gleichauf, ist die Zaehleinheit gut uebertragbar und die")
    print("Streuung eine Eigenschaft des Bildmaterials.")

    ziel = PROJEKT / "ergebnisse" / "inter_rater.csv"
    with open(ziel, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
        w.writeheader(); w.writerows(zeilen)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 6))
    hoch = max(za.max(), zb.max()) * 1.08
    ax.plot([0, hoch], [0, hoch], "--", color="#909497", linewidth=1,
            label="Gleichstand")
    ax.plot(za, zb, "o", color="#1a5276", alpha=.75, markersize=7)
    ax.set_xlabel(f"Punkte je Kachel — {args.name_a}")
    ax.set_ylabel(f"Punkte je Kachel — {args.name_b}")
    ax.set_xlim(0, hoch); ax.set_ylim(0, hoch); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"Uebereinstimmung zweier Annotierender\n"
                 f"{len(gemeinsam)} Kacheln, F1 {f1:.3f}, r {korr:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(PROJEKT / "ergebnisse" / "inter_rater.png", dpi=150)
    print(f"\nTabelle   : {ziel}")
    print(f"Abbildung : {PROJEKT/'ergebnisse'/'inter_rater.png'}")


if __name__ == "__main__":
    main()
