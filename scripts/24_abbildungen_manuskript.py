#!/usr/bin/env python3
"""Schritt 24: Abbildungen fuer das Manuskript, auf LNI-Satzbreite (12,6 cm).

Gerendert in Endgroesse, nicht skaliert: Schrift 7 pt ergibt bei Times-
Satz rund 1,7 mm Versalhoehe, ueber der LNI-Untergrenze von 1 mm.

Abb. 1  (a) Kachelausschnitt mit den Punkten beider Annotatoren
        (b) F1 bei r = 12 px ueber der gemessenen GSD (Version 2: eine Toleranz,
            unsichere Punkte als Ignorierzonen, Dateien *_ign): YOLOv8n + Punkt-NMS
            und DINOv2-Kopf (KV, 10 Seeds), Farbschwellwert (KV, 23), verblindete
            Wiederholung durch A (12 Kacheln) und zweite Person (nur nativ), beide
            aus referenz_kennzahlen_ign.json (34)

Liest nur Ergebnisdateien, keine Gewichte.

    python3 24_abbildungen_manuskript.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gemeinsam import punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
ZIEL = PROJEKT / "paper"
CM = 1 / 2.54

BEISPIEL = "DJI_20260429134535_0061_V_point12_x2060_y904.jpg"
AUSSCHNITT = (704, 200, 1024, 520)          # x0, y0, x1, y1 in Kachelpixeln


def lies(pfad: Path) -> list[dict]:
    with open(pfad, encoding="utf8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": .6,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "pdf.fonttype": 42})

    # ---------- Daten (b) ----------
    stufen = lies(ERG / "kv_bereinigt_stufen.csv")
    gsd = np.array([float(z["gsd_gemessen"]) for z in stufen])
    marken = [z["stufe"] for z in stufen]
    # erweiterte Stufen 8-20 mm/px (YOLOv8n, DINOv2, Farbschwelle), falls ausgewertet
    erw = lies(ERG / "kv_bereinigt_erw_pnms12_ign_stufen.csv") if (ERG / "kv_bereinigt_erw_pnms12_ign_stufen.csv").exists() else []
    gsd_alle = np.r_[gsd, [float(z["gsd_gemessen"]) for z in erw]]
    marken_alle = marken + [z["stufe"] for z in erw]
    # YOLO mit Punkt-NMS 12 px (Befund 2c, ohne NMS ~0,2 F1 zu niedrig)
    def laeufe(*namen):
        zl = [z for n in namen if (ERG / f"{n}_laeufe.csv").exists() for z in lies(ERG / f"{n}_laeufe.csv")]
        return {m: [float(z["f1_r12"]) for z in zl if z["stufe"] == m] for m in marken_alle}
    yolo = laeufe("kv_bereinigt_pnms12_ign", "kv_bereinigt_erw_pnms12_ign")
    dino = laeufe("kv_ssl_dinov2_s_ign", "kv_ssl_dinov2_s_erw_ign")
    dino15 = laeufe("kv_ssl_dinov2_s_x1.5_ign")
    schwelle = {z["stufe"]: float(z["baseline_f1"])
                for n in ("schwellwert_kv_ign_stufen.csv", "schwellwert_kv_erw_ign_stufen.csv")
                if (ERG / n).exists() for z in lies(ERG / n)
                if float(z["radius_px"]) == 12.0}
    ref = json.loads((ERG / "referenz_kennzahlen_ign.json").read_text(encoding="utf8"))
    mensch = np.array([ref["wiederholung_verblindet"][f"{float(z['hoehe_m']):.1f}"]["r12"]["f1"]
                       for z in stufen])
    inter_f1 = ref["zweite_person"]["r12"]["f1"]

    # ---------- Daten (a) ----------
    ordner = [d for d in sorted((PROJEKT / "kacheln").glob("*")) if d.is_dir()][0]
    bild = Image.open(ordner / BEISPIEL).convert("RGB").crop(AUSSCHNITT)
    a = np.array(punkte_aus_export(PROJEKT / "annotation" / "export.json")
                 [BEISPIEL]["punkte"]) * 1024 / 100
    b = np.array(punkte_aus_export(PROJEKT / "annotation" / "export_annotator2.json")
                 [BEISPIEL]["punkte"]).reshape(-1, 2) * 1024 / 100
    x0, y0, x1, y1 = AUSSCHNITT

    def im_ausschnitt(p):
        m = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
        return p[m] - [x0, y0]

    a_s, b_s = im_ausschnitt(a), im_ausschnitt(b)

    fig = plt.figure(figsize=(12.6 * CM, 5.2 * CM))
    ax1 = fig.add_axes([0.0, 0.03, 0.40, 0.94])
    ax2 = fig.add_axes([0.505, 0.17, 0.475, 0.80])

    ax1.imshow(bild)
    ax1.scatter(a_s[:, 0], a_s[:, 1], s=26, facecolors="none", edgecolors="#ff3b30",
                linewidths=.9, label=f"annotator A ({len(a_s)})")
    ax1.scatter(b_s[:, 0], b_s[:, 1], s=9, marker="s", facecolors="none",
                edgecolors="#2fa8ff", linewidths=.9, label=f"annotator B ({len(b_s)})")
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.legend(loc="lower left", frameon=True, framealpha=.85, borderpad=.3,
               handletextpad=.2, borderaxespad=.3)
    mm = (x1 - x0) * 1.714
    ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", ha="left",
             fontsize=7.5, fontweight="bold", color="white")
    ax1.plot([x1 - x0 - 12 - 100 / 1.714, x1 - x0 - 12], [18, 18], color="white", lw=1.6)
    ax1.text(x1 - x0 - 12 - 50 / 1.714, 28, "10 cm", color="white", ha="center",
             va="top", fontsize=6.5)

    # (b)
    for g, m in zip(gsd_alle, marken_alle):
        for werte, farbe, dx in ((yolo, "#7f7f7f", .985), (dino, "#7fa6d6", 1.0), (dino15, "#1f3f73", 1.015)):
            if werte[m]:
                ax2.plot(np.full(len(werte[m]), g * dx), werte[m], "o", ms=1.6, color=farbe,
                         alpha=.5, mew=0, zorder=2)
    def linie(werte, **kw):
        ms = [m for m in marken_alle if werte[m]]
        ax2.plot([gsd_alle[marken_alle.index(m)] for m in ms], [np.mean(werte[m]) for m in ms], **kw)
    linie(dino15, ls="-", color="#1f3f73", lw=1.1, label="DINOv2 ×1.5", zorder=3)
    linie(dino, ls="-", color="#7fa6d6", lw=1.1, label="DINOv2", zorder=3)
    linie(yolo, ls="-", color="#333333", lw=1.1, label="YOLOv8n", zorder=3)
    ms = [m for m in marken_alle if m in schwelle]
    ax2.plot([gsd_alle[marken_alle.index(m)] for m in ms], [schwelle[m] for m in ms], "--",
             color="#a8732b", lw=1.1, marker="D", ms=2.4, label="threshold", zorder=3)
    ax2.plot(gsd, mensch, "-", color="#3f6b4a", lw=1.1, marker="s", ms=2.8,
             label="A, blinded repeat", zorder=3)
    ax2.plot([gsd[0]], [inter_f1], marker="^", ms=4.2, color="#9c3b2e", ls="none",
             label="B (native only)", zorder=4)
    ax2.set_xlabel("ground sampling distance at ground level (mm/px)")
    ax2.set_ylabel("F1 against annotator A")
    ax2.set_xscale("log")
    ax2.set_xlim(1.5, 23); ax2.set_ylim(0.3, 0.85)
    ax2.set_xticks([1.7, 3, 4, 5, 8, 11, 15, 20])
    ax2.set_xticklabels(["1.7", "3", "4", "5", "8", "11", "15", "20"])
    ax2.minorticks_off()
    ax2.grid(alpha=.25, lw=.4)
    ax2.legend(loc="lower right", bbox_to_anchor=(0.80, 0.0), frameon=False, handlelength=2.0,
               borderaxespad=.2, ncol=2, columnspacing=0.8, fontsize=6)
    ax2.text(-0.2, 1.0, "(b)", transform=ax2.transAxes, va="top", ha="left",
             fontsize=7.5, fontweight="bold")

    ZIEL.mkdir(exist_ok=True)
    for endung in ("png", "pdf"):
        fig.savefig(ZIEL / f"abb1.{endung}", dpi=600)
    print(f"-> {ZIEL / 'abb1.png'}  (Ausschnitt {mm/10:.0f} cm breit, "
          f"A {len(a_s)} / B {len(b_s)} Punkte)")


if __name__ == "__main__":
    main()
