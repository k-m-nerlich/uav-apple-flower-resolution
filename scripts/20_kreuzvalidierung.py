#!/usr/bin/env python3
"""Schritt 20: Kreuzvalidierung ueber Aufloesungsstufen und Trainingsseeds.

Warum
-----
07/08 bewerten auf EINER festen Aufteilung (26 train / 10 val) und benutzen
die 10 Bewertungskacheln zugleich fuer
  * die Wahl von best.pt (ultralytics validiert jede Epoche auf val) und
  * die Wahl der Konfidenzschwelle (bestes F1 im Sweep).
Beides schoent das Ergebnis, und die Streuung durch die Wahl der Aufteilung
steckt in keiner Zahl.

Aufbau
------
36 Kacheln, einmal mit Seed 42 gemischt, reihum auf 6 Falten verteilt.
Fuer Falte k:
    test  = Falte k              (6 Kacheln, nie gesehen)
    val   = Falte (k+1) mod 6    (6 Kacheln, waehlt best.pt und die Schwelle)
    train = die uebrigen vier    (24 Kacheln)
Jede Kachel ist damit genau einmal Testkachel. Je Stufe und Seed entsteht ein
F1 ueber alle 36 Kacheln. Training sonst identisch zu 07_pro_stufe.py.

Die Stufen werden wie in 07 aus den nativen Kacheln heruntergerechnet
(LANCZOS herunter, BICUBIC zurueck auf 1024 px). Annotation ist die native.

Ausgabe
-------
Dieses Skript trainiert nur und speichert Rohvorhersagen (ab conf 0,01) fuer
die val- und test-Kacheln nach ergebnisse/kv/vorhersagen/. Alles Weitere -
Schwellenwahl, Radien, AP, Zaehlfehler, Bootstrap - rechnet
21_kv_auswertung.py daraus, ohne GPU.

Wiederaufnehmbar: Laeufe mit vorhandener Vorhersagedatei werden uebersprungen.
Mehrere Prozesse parallel sind moeglich, solange sie verschiedene Stufen
bearbeiten (jede Stufe hat ihren eigenen Datensatzordner).

    python 20_kreuzvalidierung.py --stufen 5.1 --seeds 0 1 2 --device 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from gemeinsam import mm_pro_pixel, punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent
STUFEN_M = [5.1, 8.8, 10.7, 14.8, 16.7]
# gemessen (13_geometrie_kalibrieren.py): 0,3418 * 5,1 m - 0,029
NATIVE_GSD_GEMESSEN = 1.714


def marke_von(hoehe: float) -> str:
    return f"h{round(hoehe * 10):03d}"


def falten_bilden(namen: list[str], anzahl: int, seed: int) -> list[list[str]]:
    namen = sorted(namen)
    random.Random(seed).shuffle(namen)
    return [sorted(namen[i::anzahl]) for i in range(anzahl)]


def boxgroessen(punkte: dict, pfad: Path | None, box_px: float) -> dict:
    """Boxkante je Punkt in px. Ohne Groessendatei: ueberall box_px (Stand bis
    21.10.2026). Mit Groessendatei (31_bluetengroesse.py): relativ skaliert,
    box_px * d / median(d), begrenzt auf [box_px/2, 2*box_px] - mittlere
    Boxgroesse bleibt gleich, nur die Variation je Bluete kommt hinzu."""
    if pfad is None:
        return {n: [box_px] * len(p) for n, p in punkte.items()}
    g = json.loads(pfad.read_text(encoding="utf8"))["groessen"]
    med = float(np.median([v for n in punkte for v in g[n]]))
    aus = {}
    for n, p in punkte.items():
        if len(g[n]) != len(p):
            raise SystemExit(f"Groessendatei passt nicht zu {n}: {len(g[n])} gegen {len(p)} Punkte")
        aus[n] = [float(np.clip(box_px * v / med, box_px / 2, box_px * 2)) for v in g[n]]
    return aus


def baue_stufe(punkte: dict, kachelordner: Path, ziel: Path, faktor: float,
               boxen: dict, kachel_px: int) -> None:
    """Alle Kacheln einer Stufe einmal degradieren (wie 07, ohne Aufteilung)."""
    fertig = ziel / "fertig.txt"
    if fertig.exists():
        return
    (ziel / "images").mkdir(parents=True, exist_ok=True)
    (ziel / "labels").mkdir(parents=True, exist_ok=True)
    for name, pkt in punkte.items():
        bild = Image.open(kachelordner / name).convert("RGB")
        if faktor < 0.999:
            klein = max(32, int(round(kachel_px * faktor)))
            bild = (bild.resize((klein, klein), Image.LANCZOS)
                        .resize((kachel_px, kachel_px), Image.BICUBIC))
        bild.save(ziel / "images" / name, quality=95)
        (ziel / "labels" / f"{Path(name).stem}.txt").write_text(
            "\n".join(f"0 {x/100:.6f} {y/100:.6f} {b/kachel_px:.6f} {b/kachel_px:.6f}"
                      for (x, y), b in zip(pkt, boxen[name])), encoding="utf8")
    fertig.write_text("ok", encoding="utf8")


def teile_falte(k: int, falten: list[list[str]],
                sperren: dict[str, set[str]] | None = None) -> dict:
    """test = Falte k, val = Falte k+1, train = Rest. Mit `sperren` werden aus
    val und train alle Kacheln entfernt, die sich am Boden mit einer
    Testkachel ueberlappen (purged CV)."""
    n = len(falten)
    teile = {"test": falten[k], "val": falten[(k + 1) % n],
             "train": [x for i, f in enumerate(falten)
                       if i not in (k, (k + 1) % n) for x in f]}
    if sperren:
        weg = set().union(*(sperren.get(x, set()) for x in teile["test"]))
        teile["val"] = [x for x in teile["val"] if x not in weg]
        teile["train"] = [x for x in teile["train"] if x not in weg]
    return teile


def schreibe_falte(ziel: Path, k: int, teile: dict, praefix: str) -> Path:
    for teil, namen in teile.items():
        (ziel / f"{praefix}falte{k}_{teil}.txt").write_text(
            "\n".join(str(ziel / "images" / x) for x in namen) + "\n",
            encoding="utf8")
    yaml_pfad = ziel / f"{praefix}falte{k}.yaml"
    yaml_pfad.write_text(yaml.safe_dump({
        "path": str(ziel), "train": f"{praefix}falte{k}_train.txt",
        "val": f"{praefix}falte{k}_val.txt", "names": {0: "Bluete"},
    }, sort_keys=False), encoding="utf8")
    return yaml_pfad


def vorhersagen(modell, bilder: list[Path], imgsz: int, kachel_px: int,
                device: str, max_det: int) -> dict:
    aus = {}
    for pfad in bilder:
        r = modell.predict(Image.open(pfad).convert("RGB"), conf=0.01,
                           imgsz=imgsz, max_det=max_det, device=device,
                           verbose=False)[0]
        skala = kachel_px / imgsz
        if len(r.boxes):
            xy = r.boxes.xywh.cpu().numpy()[:, :2] * skala
            c = r.boxes.conf.cpu().numpy()
            aus[pfad.name] = [[round(float(a), 1), round(float(b), 1),
                               round(float(s), 4)] for (a, b), s in zip(xy, c)]
        else:
            aus[pfad.name] = []
    return aus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path,
                   default=PROJEKT / "annotation" / "export.json")
    p.add_argument("--kachelordner", type=Path, default=None)
    p.add_argument("--stufen", type=float, nargs="+", default=STUFEN_M)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--falten", type=int, default=6)
    p.add_argument("--nur-falten", type=int, nargs="+", default=None,
                   help="nur diese Falten rechnen (zum Testen)")
    p.add_argument("--aufteilungsseed", type=int, default=42)
    p.add_argument("--box-px", type=int, default=24)
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochen", type=int, default=100)
    p.add_argument("--modell", default="yolov8n.pt")
    p.add_argument("--device", default="mps")
    p.add_argument("--max-det-vorhersage", type=int, default=5000,
                   help="Obergrenze nur fuer die gespeicherten Rohvorhersagen. "
                        "1200 (wie im Training) kappt bei breit feuernden "
                        "Modellen schon oberhalb conf 0,05.")
    p.add_argument("--ausgabe", default="kv",
                   help="Ordner unter ergebnisse/ (kv = erste, ungesperrte KV)")
    p.add_argument("--sperrliste", type=Path, default=None,
                   help="JSON mit 'kanten' [[inlier, a, b], ...] ueberlappender "
                        "Kacheln (SIFT-Pruefung); diese werden je Falte aus "
                        "train/val entfernt, wenn der Partner Testkachel ist")
    p.add_argument("--nur-vorhersagen", action="store_true",
                   help="nicht trainieren, nur aus vorhandenen best.pt die "
                        "Vorhersagedateien neu schreiben")
    p.add_argument("--groessen", type=Path, default=None,
                   help="annotation/punktgroessen.json (31): Boxgroesse je Bluete "
                        "statt fester --box-px; Datensaetze dann unter <stufe>_gr")
    args = p.parse_args()

    kachelordner = args.kachelordner
    if kachelordner is None:
        unter = [d for d in sorted((PROJEKT / "kacheln").glob("*")) if d.is_dir()]
        if not unter:
            raise SystemExit("Keine Kacheln gefunden.")
        kachelordner = unter[0]

    roh = punkte_aus_export(args.export)
    punkte = {n: e["punkte"] for n, e in roh.items() if (kachelordner / n).exists()}
    falten = falten_bilden(list(punkte), args.falten, args.aufteilungsseed)
    sperren: dict[str, set[str]] = {}
    if args.sperrliste:
        for _, a, b in json.loads(args.sperrliste.read_text(encoding="utf8"))["kanten"]:
            sperren.setdefault(a, set()).add(b)
            sperren.setdefault(b, set()).add(a)
    teilungen = [teile_falte(k, falten, sperren) for k in range(args.falten)]

    kv = PROJEKT / "ergebnisse" / args.ausgabe
    praefix = "" if args.ausgabe == "kv" else f"{args.ausgabe}_"
    (kv / "vorhersagen").mkdir(parents=True, exist_ok=True)
    (kv / "falten.json").write_text(json.dumps(
        {"aufteilungsseed": args.aufteilungsseed, "falten": falten,
         "sperrliste": str(args.sperrliste) if args.sperrliste else None,
         "teilungen": teilungen}, indent=1), encoding="utf8")
    for k, t in enumerate(teilungen):
        print(f"Falte {k}: train {len(t['train'])}, val {len(t['val'])}, "
              f"test {len(t['test'])}", flush=True)

    from ultralytics import YOLO

    basis = mm_pro_pixel(STUFEN_M[0])
    for hoehe in args.stufen:
        faktor = basis / mm_pro_pixel(hoehe)
        marke = marke_von(hoehe)
        datensatz = PROJEKT / "datensaetze_kv" / (marke + ("_gr" if args.groessen else ""))
        baue_stufe(punkte, kachelordner, datensatz, faktor,
                   boxgroessen(punkte, args.groessen, args.box_px), args.kachel_px)

        for seed in args.seeds:
            for k in (args.nur_falten or range(args.falten)):
                lauf = f"{marke}_f{k}_s{seed}"
                ziel = kv / "vorhersagen" / f"{lauf}.json"
                gew = kv / "laeufe" / marke / f"f{k}_s{seed}" / "weights"
                t0 = time.time()
                if args.nur_vorhersagen:
                    if not (gew / "best.pt").exists():
                        print(f"### {lauf}: keine Gewichte, uebersprungen")
                        continue
                    if ziel.exists() and json.loads(ziel.read_text(
                            encoding="utf8")).get("max_det") == args.max_det_vorhersage:
                        continue
                else:
                    if ziel.exists():
                        continue
                    yaml_pfad = schreibe_falte(datensatz, k, teilungen[k], praefix)
                    modell = YOLO(args.modell)
                    modell.train(data=str(yaml_pfad), epochs=args.epochen,
                                 imgsz=args.imgsz, device=args.device, batch=4,
                                 workers=2, project=str(kv / "laeufe" / marke),
                                 name=f"f{k}_s{seed}", exist_ok=True, max_det=1200,
                                 hsv_h=0.010, hsv_s=0.4, hsv_v=0.3,
                                 degrees=180, fliplr=0.5, flipud=0.5, mosaic=0.5,
                                 seed=seed, plots=False, verbose=False)
                bester = YOLO(str(gew / "best.pt"))
                bilder = lambda teil: [datensatz / "images" / x for x in teil]
                daten = {
                    "hoehe_m": hoehe, "faktor": round(faktor, 4),
                    "gsd_nominell": round(mm_pro_pixel(hoehe), 3),
                    "gsd_gemessen": round(NATIVE_GSD_GEMESSEN / faktor, 3),
                    "falte": k, "seed": seed, "max_det": args.max_det_vorhersage,
                    "test": vorhersagen(bester, bilder(teilungen[k]["test"]), args.imgsz,
                                        args.kachel_px, args.device,
                                        args.max_det_vorhersage),
                    "val": vorhersagen(bester, bilder(teilungen[k]["val"]),
                                       args.imgsz, args.kachel_px, args.device,
                                       args.max_det_vorhersage),
                }
                ziel.write_text(json.dumps(daten), encoding="utf8")
                (gew / "last.pt").unlink(missing_ok=True)
                print(f"### {lauf} fertig in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
