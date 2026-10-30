#!/usr/bin/env python3
"""Schritt 6: Die menschliche Decke auswerten.

Wertet den Mehrstufen-Export gegen die Referenzannotation der nativen Stufe aus
und liefert drei Kurven ueber die Bodenaufloesung:

  Wiederfindung  Anteil der Referenzblueten, die auf der degradierten Stufe
                 noch markiert wurden. Die eigentliche Aussage.
  Zusaetzliche   Markierungen ohne Entsprechung in der Referenz. Bei groben
                 Stufen sind das teils zusammengefasste Nachbarblueten, teils
                 Fehldeutungen von hellem Boden.
  Unsicherheit   Anteil der als "unsicher" gesetzten Punkte. Steigt er stark
                 an, ist die Aufgabe bei dieser Aufloesung nicht mehr
                 eindeutig loesbar - unabhaengig von jedem Modell.

Zusaetzlich wird aus der wiederholten nativen Stufe die
Intra-Rater-Uebereinstimmung berechnet.

    python3 06_menschliche_decke.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from gemeinsam import punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent


def zuordnen(referenz: np.ndarray, neu: np.ndarray, radius: float):
    """Greedy-Zuordnung nach Distanz. Rueckgabe (TP, zusaetzlich, verfehlt)."""
    if len(referenz) == 0:
        return 0, len(neu), 0
    if len(neu) == 0:
        return 0, 0, len(referenz)
    d = np.linalg.norm(referenz[:, None, :] - neu[None, :, :], axis=2)
    paare = np.argwhere(d <= radius)
    frei_r = np.ones(len(referenz), bool)
    frei_n = np.ones(len(neu), bool)
    tp = 0
    for ir, ineu in sorted(paare.tolist(), key=lambda p: d[p[0], p[1]]):
        if frei_r[ir] and frei_n[ineu]:
            frei_r[ir] = frei_n[ineu] = False
            tp += 1
    return tp, int(frei_n.sum()), int(frei_r.sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--referenz", type=Path,
                   default=PROJEKT / "annotation" / "export.json")
    p.add_argument("--mehrstufig", type=Path,
                   default=PROJEKT / "annotation" / "export_mehrstufig.json")
    p.add_argument("--schluessel", type=Path,
                   default=PROJEKT / "kacheln_mehrstufig" / "schluessel.csv")
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--radius-px", type=float, default=16.0,
                   help="Zuordnungsradius; grosszuegiger als beim Modell, weil "
                        "Menschen den Punkt unterschiedlich setzen")
    args = p.parse_args()

    for pfad, was in ((args.referenz, "Referenzannotation"),
                      (args.mehrstufig, "Mehrstufen-Export"),
                      (args.schluessel, "Schluesseldatei")):
        if not pfad.exists():
            raise SystemExit(f"{was} nicht gefunden: {pfad}")

    referenz = punkte_aus_export(args.referenz)
    neu = punkte_aus_export(args.mehrstufig)
    with open(args.schluessel, encoding="utf8") as f:
        schluessel = list(csv.DictReader(f))

    K = args.kachel_px
    proz = lambda liste: (np.array(liste) / 100.0 * K if liste
                          else np.zeros((0, 2)))

    je_stufe = defaultdict(lambda: {"tp": 0, "zus": 0, "fehlt": 0,
                                    "unsicher": 0, "gesetzt": 0, "kacheln": 0})
    fehlend = 0
    for eintrag in schluessel:
        anonym, quelle = eintrag["anonym"], eintrag["kachel"]
        if anonym not in neu:
            fehlend += 1
            continue
        r = proz(referenz.get(quelle, {}).get("punkte", []))
        n = proz(neu[anonym]["punkte"])
        tp, zus, fehlt = zuordnen(r, n, args.radius_px)

        s = je_stufe[(float(eintrag["hoehe_m"]), float(eintrag["mm_pro_px"]))]
        s["tp"] += tp; s["zus"] += zus; s["fehlt"] += fehlt
        s["unsicher"] += neu[anonym]["unsicher"]
        s["gesetzt"] += len(n)
        s["kacheln"] += 1

    if not je_stufe:
        raise SystemExit("Keine Zuordnung moeglich - passen Schluessel und "
                         "Export zusammen?")
    if fehlend:
        print(f"WARNUNG   : {fehlend} Aufgaben aus dem Schluessel fehlen im "
              f"Export (noch nicht annotiert?)\n")

    zeilen = []
    for (hoehe, gsd) in sorted(je_stufe, key=lambda k: k[1]):
        s = je_stufe[(hoehe, gsd)]
        bezug = s["tp"] + s["fehlt"]
        zeilen.append({
            "hoehe_m": hoehe, "mm_pro_px": gsd, "kacheln": s["kacheln"],
            "referenzpunkte": bezug, "gesetzt": s["gesetzt"],
            "wiedergefunden": s["tp"], "zusaetzlich": s["zus"],
            "verfehlt": s["fehlt"],
            "wiederfindung": round(s["tp"] / bezug, 3) if bezug else 0.0,
            "zusatzquote": round(s["zus"] / s["gesetzt"], 3) if s["gesetzt"] else 0.0,
            "unsicherheitsquote": round(
                s["unsicher"] / (s["gesetzt"] + s["unsicher"]), 3)
            if s["gesetzt"] + s["unsicher"] else 0.0,
        })

    print(f"{'mm/px':>7} {'Hoehe':>7} {'Kach':>5} {'Ref':>6} {'gefunden':>9} "
          f"{'Wiederfind.':>12} {'zusaetzl.':>10} {'unsicher':>9}")
    for z in zeilen:
        print(f"{z['mm_pro_px']:7.2f} {z['hoehe_m']:6.1f}m {z['kacheln']:5d} "
              f"{z['referenzpunkte']:6d} {z['wiedergefunden']:9d} "
              f"{z['wiederfindung']:12.3f} {z['zusatzquote']:10.3f} "
              f"{z['unsicherheitsquote']:9.3f}")

    basis = zeilen[0]
    print(f"\nIntra-Rater (native Stufe, {basis['mm_pro_px']:.2f} mm/px, "
          f"Wiederholung nach Runde 1):")
    print(f"  Wiederfindung {basis['wiederfindung']:.3f}, "
          f"Zusatzquote {basis['zusatzquote']:.3f}")
    print("  Das ist die Obergrenze dessen, was jede spaetere Messung erreichen "
          "kann.\n  Alle Werte der groeberen Stufen sind daran zu relativieren.")

    csv_pfad = PROJEKT / "ergebnisse" / "menschliche_decke.csv"
    with open(csv_pfad, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0]))
        w.writeheader(); w.writerows(zeilen)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [z["mm_pro_px"] for z in zeilen]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(g, [z["wiederfindung"] for z in zeilen], "o-", color="#1a5276",
            linewidth=2.4, label="Wiederfindung (Mensch)")
    ax.plot(g, [z["zusatzquote"] for z in zeilen], "s--", color="#b9770e",
            label="zusaetzliche Markierungen")
    ax.plot(g, [z["unsicherheitsquote"] for z in zeilen], "^--", color="#7d3c98",
            label="als 'unsicher' markiert")
    ax.axhline(basis["wiederfindung"], color="#909497", linestyle=":",
               label=f"Intra-Rater-Obergrenze ({basis['wiederfindung']:.2f})")
    ax.set_xlabel("Bodenaufloesung (mm/px)")
    ax.set_ylabel("Anteil")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(loc="center left")
    ax.set_title("Menschliche Decke der Bluetenerkennung\n"
                 f"Ablass Elstar, {basis['kacheln']} Kacheln je Stufe, "
                 "Referenz aus der feinsten Stufe", fontsize=11)
    fig.tight_layout()
    plot = PROJEKT / "ergebnisse" / "menschliche_decke.png"
    fig.savefig(plot, dpi=150)

    print(f"\nTabelle   : {csv_pfad}")
    print(f"Abbildung : {plot}")


if __name__ == "__main__":
    main()
