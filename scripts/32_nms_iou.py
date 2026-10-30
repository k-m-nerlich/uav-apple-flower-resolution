#!/usr/bin/env python3
"""Schritt 32: YOLO mit abgestimmter Box-NMS statt Punkt-Merging (Review 3.6).

Gutachten: Ohne Merging erreicht YOLO nur 0,49-0,51; die uebliche Abhilfe waere
eine niedrigere IoU-Schwelle der Box-NMS. Hier wird sie geprueft, ohne
Punkt-Merging:

* Gewichte der bereinigten KV (ergebnisse/kv_bereinigt/laeufe/<stufe>/f<k>_s<seed>)
* Vorhersage mit IoU in IOU_GITTER (ultralytics-Voreinstellung 0,7)
* je Lauf die IoU waehlen, die auf der VALIDIERUNGSfalte das beste F1 ergibt
  (ueber das Konfidenzgitter von 21, r = 12 px)
* Rohvorhersagen der gewaehlten IoU im Format von 20 nach
  ergebnisse/kv_bereinigt_iouval/vorhersagen schreiben -> Auswertung mit 21:

      python 21_kv_auswertung.py --kv kv_bereinigt_iouval

Je Lauf landen die val-F1 aller IoU-Werte in iou_wahl.json.

    python 32_nms_iou.py --device 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
IOU_GITTER = [0.1, 0.2, 0.3, 0.5, 0.7]


def modul(datei):
    spec = importlib.util.spec_from_file_location(datei[:2], Path(__file__).with_name(datei))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quelle", default="kv_bereinigt")
    ap.add_argument("--ziel", default="kv_bereinigt_iouval")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=5000)
    args = ap.parse_args()

    from ultralytics import YOLO
    from gemeinsam import punkte_aus_export
    kv21 = modul("21_kv_auswertung.py")

    quelle, ziel = ERG / args.quelle, ERG / args.ziel
    (ziel / "vorhersagen").mkdir(parents=True, exist_ok=True)
    shutil.copy(quelle / "falten.json", ziel / "falten.json")
    roh = punkte_aus_export(PROJEKT / "annotation" / "export.json")
    ref = {k: np.array(v["punkte"], float).reshape(-1, 2) * 10.24 for k, v in roh.items()}
    wahl_pfad = ziel / "iou_wahl.json"
    wahl = json.loads(wahl_pfad.read_text(encoding="utf8")) if wahl_pfad.exists() else {}

    for vf in sorted((quelle / "vorhersagen").glob("*.json")):
        aus = ziel / "vorhersagen" / vf.name
        if aus.exists():
            continue
        alt = json.loads(vf.read_text(encoding="utf8"))
        stufe, f, s = vf.stem.split("_")
        gew = quelle / "laeufe" / stufe / f"{f}_{s}" / "weights" / "best.pt"
        modell = YOLO(str(gew))
        ordner = PROJEKT / "datensaetze_kv" / stufe / "images"
        bilder = {t: {k: Image.open(ordner / k).convert("RGB") for k in alt[t]} for t in ("val", "test")}
        je_iou, val_f1 = {}, {}
        for iou in IOU_GITTER:
            v = {}
            for t in ("val", "test"):
                v[t] = {}
                for k, b in bilder[t].items():
                    r = modell.predict(b, conf=0.01, iou=iou, imgsz=1024, max_det=args.max_det,
                                       device=args.device, verbose=False)[0]
                    if len(r.boxes):
                        xy = r.boxes.xywh.cpu().numpy()[:, :2]; c = r.boxes.conf.cpu().numpy()
                        v[t][k] = [[round(float(a), 1), round(float(bb), 1), round(float(cc), 4)]
                                   for (a, bb), cc in zip(xy, c)]
                    else:
                        v[t][k] = []
            je_iou[iou] = v
            beste = 0.0
            for conf in kv21.CONF_GITTER:
                summe = np.zeros(3)
                for k, d in v["val"].items():
                    a = np.array(d, float).reshape(-1, 3)
                    summe += kv21.zuordnen(ref[k], a[a[:, 2] >= conf, :2], 12.0)
                beste = max(beste, kv21.f1(*summe))
            val_f1[iou] = beste
        gewaehlt = max(IOU_GITTER, key=lambda i: (val_f1[i], -i))
        daten = {k: alt[k] for k in alt if k not in ("test", "val")}
        daten.update({"iou": gewaehlt, "max_det": args.max_det, **je_iou[gewaehlt]})
        aus.write_text(json.dumps(daten), encoding="utf8")
        wahl[vf.stem] = {"gewaehlt": gewaehlt, "val_f1": {str(i): round(x, 4) for i, x in val_f1.items()}}
        wahl_pfad.write_text(json.dumps(wahl, indent=1), encoding="utf8")
        print(f"### {vf.stem}: IoU {gewaehlt} (val-F1 " +
              ", ".join(f"{i}: {x:.3f}" for i, x in val_f1.items()) + ")", flush=True)


if __name__ == "__main__":
    main()
