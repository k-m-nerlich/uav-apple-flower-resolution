# The reference is the bottleneck in UAV-based apple flower detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23806339.svg)](https://doi.org/10.5281/zenodo.23806339)

Data, annotations, raw predictions and code for

> Nerlich, K. M.: The reference is the bottleneck in UAV-based apple flower detection. 47. GIL-Jahrestagung 2027,
> Lecture Notes in Informatics (LNI), Gesellschaft für Informatik, Bonn, 2027.

The study compares the effect of simulated ground resolution (1.7 to 20 mm/px)
on apple flower detection with training variability (6 folds × 10 seeds), the
choice of method (YOLOv8n/s, frozen DINOv2-small and I-JEPA ViT-H/14 encoders
with a heatmap head, colour threshold) and the reliability of the manual
reference (blinded repeat, repeat after 53 days, second annotator).

Script names, file names, column names and comments are in German; this README
gives the mapping to the paper.

## Main results (F1 at 12 px = 21 mm, purged cross-validation, uncertain points as ignore zones)

| Ground GSD (mm/px) | 1.71 | 2.96 | 3.60 | 4.97 | 5.61 | 8.0 | 11.0 | 15.0 | 20.0 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n, duplicates merged | 0.691 | 0.689 | 0.693 | 0.695 | 0.698 | 0.682 | 0.662 | 0.651 | 0.612 |
| DINOv2-small head | 0.719 | 0.712 | 0.714 | 0.709 | 0.711 | 0.700 | 0.680 | 0.615 | 0.570 |
| DINOv2-small head, input ×1.5 | 0.734 | 0.729 | 0.733 | 0.730 | 0.728 | – | – | – | – |
| Colour threshold | 0.516 | 0.532 | 0.526 | 0.541 | 0.541 | 0.553 | 0.539 | 0.466 | 0.403 |
| Annotator A, blinded repeat (12 tiles) | 0.781 | 0.766 | 0.750 | 0.720 | 0.711 | – | – | – | – |

Annotator A against a repeat after 53 days: 0.741. Second annotator B against A:
0.364 (precision 0.92, recall 0.23).

## Repository structure

```
annotation/            Label Studio exports (JSON) and annotation protocol
  export.json            annotator A, round 1 (reference, 36 tiles, 1,714 points)
  export_runde3.json     annotator A, repeat after 53 days
  export_annotator2.json annotator B
  export_mehrstufig.json annotator A, blinded repeat: 12 tiles × 5 levels (k0001–k0060)
  export_zauberwald.json annotator A, site B (29 tiles, real 5.44 mm/px)
  punktgroessen.json     flower size per point (white area, script 31)
  ANNOTATIONSPROTOKOLL.md written protocol given to annotator B (German; English: docs/)
kacheln/               36 native tiles, 1024 × 1024 px, 1.71 mm/px (site A, 5.1 m), manifest.csv
kacheln_mehrstufig/    60 anonymised degraded tiles for the blinded repeat, schluessel.csv = key
kacheln_zauberwald/    29 tiles of site B, manifest.csv
ergebnisse/            all evaluated results (CSV/JSON/PNG)
  kv/falten.json         folds (test = k, validation = k + 1)
  kv/kachelgruppen.json  16 groups of overlapping tiles (purging, cluster bootstrap)
  <variant>/vorhersagen.zip  raw predictions of every run (conf ≥ 0.01), see below
scripts/               analysis code (Python 3.12)
docs/                  checked studies, English annotation protocol
```

Variants (`ergebnisse/<variant>/`):

| Folder | Content |
|---|---|
| `kv_bereinigt` | YOLOv8n, purged CV, 5 levels × 6 folds × 10 seeds |
| `kv` | YOLOv8n, CV without purging (leakage check) |
| `kv_bereinigt_iouval` | YOLOv8n, box-NMS IoU tuned on validation (`iou_wahl.json`) |
| `kv_bereinigt_v8s` | YOLOv8s control |
| `kv_bereinigt_erw` | YOLOv8n, extended levels 8 / 11 / 15 / 20 mm/px |
| `kv_bereinigt_gr` | YOLOv8n with box size per flower |
| `kv_ssl_dinov2_s`, `_x1.5`, `_erw`, `_gr` | DINOv2-small heatmap head (native input, ×1.5, extended levels, size per flower) |
| `kv_ssl_ijepa_h14_1k`, `_x1.5` | I-JEPA ViT-H/14 heatmap head |

Level names: `h051` … `h167` = simulated flight altitude in dm (5.1 to 16.7 m;
1.71 to 5.61 mm/px measured ground GSD), `h238` … `h595` = 8 to 20 mm/px.

## Where the numbers in the paper come from

| Paper | File in `ergebnisse/` | Script |
|---|---|---|
| Table 1, YOLOv8n | `kv_bereinigt_pnms12_ign_stufen.csv`, `kv_bereinigt_erw_pnms12_ign_stufen.csv` | 20 → 21 `--punkt-nms 12 --ignoriere-unsicher` |
| Table 1, DINOv2 (×1.5) | `kv_ssl_dinov2_s_ign_stufen.csv`, `kv_ssl_dinov2_s_x1.5_ign_stufen.csv`, `kv_ssl_dinov2_s_erw_ign_stufen.csv` | 25 → 21 `--ignoriere-unsicher` |
| Table 1, colour threshold | `schwellwert_kv_ign_stufen.csv`, `schwellwert_kv_erw_ign_stufen.csv` | 23 |
| Table 1, repeat A; reliability section | `referenz_kennzahlen_ign.json`, `menschliche_decke.csv` | 34, 21 |
| Slopes, equivalence | `*_ign_statistik.json` | 21 |
| Extended range vs. native | `erweiterter_bereich.csv` | 35 |
| Paired method differences | `methodenvergleich_ign.csv` | 29 `--ign` |
| Tuned NMS IoU | `kv_bereinigt_iouval_ign_*`, `kv_bereinigt_iouval/iou_wahl.json` | 32 → 21 |
| YOLOv8s | `kv_bereinigt_v8s_pnms12_ign_*` | 20 `--modell yolov8s.pt` → 21 |
| Leakage | `kv_pnms12_ign_*` vs. `kv_bereinigt_pnms12_ign_*` | 21 |
| YOLOv8n without merging | `kv_bereinigt_ign_*` | 21 |
| Transfer to site B | `uebertragbarkeit_kv.json/.csv` | 28 |
| GSD calibration | `geometrie_kalibrierung.csv` | 13 |
| Fig. 1 | – | 24 |

## Reproducing

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
cd scripts
python entpacken.py                              # unzips ergebnisse/*/vorhersagen.zip
```

Evaluation from the raw predictions (CPU only, a few minutes each):

```bash
python 21_kv_auswertung.py --kv kv_bereinigt --punkt-nms 12 --ignoriere-unsicher
python 21_kv_auswertung.py --kv kv_ssl_dinov2_s_x1.5 --ignoriere-unsicher
python 23_schwellwert_kv.py --ignoriere-unsicher --detektor-suffix _pnms12_ign
python 29_methodenvergleich.py --ign
python 34_referenz_kennzahlen.py --ignoriere-unsicher
python 35_erweiterter_bereich.py
python 24_abbildungen_manuskript.py
```

Training (GPU; about 1 min per run for the heatmap heads, longer for YOLO):

```bash
bash laeufe_kv_pc.sh kv_bereinigt --sperrliste ../ergebnisse/kv/kachelgruppen.json
python 25_ssl_kv.py --encoder dinov2_s --skala 1.5 --device cuda
```

`laeufe_kv_pc.sh` and `laeufe_version2.sh` call `../.venv/Scripts/python.exe`
(Windows); on Linux/macOS change `PY` to `../.venv/bin/python`. Pretrained weights
(COCO for YOLOv8, DINOv2 and I-JEPA from Hugging Face) are downloaded on first
use. Trained weights are not included; they can be regenerated with the
commands above. Scripts 01, 13 and 28 need the original drone images (54 GB),
which are not part of this repository and are available from the author on
request.

Notes: `FOV_FAKTOR` in `gemeinsam.py` stays at the nominal 1.50 for compatibility
with earlier runs; all GSD values in the paper use the measured calibration
(`GSD = 0.3418 · h − 0.029 mm/px`, ground plane). Repeated GPU runs with the same
seed are not bit-identical across platforms.

## Licence

Code (`scripts/`): MIT, see `LICENSE`. Data (annotations, tiles, results,
predictions): CC BY 4.0, see `LICENSE-DATA.md`. The code uses Ultralytics YOLOv8,
which is licensed under AGPL-3.0.

## Citation

See `CITATION.cff`. Please cite the paper above and this repository:
https://doi.org/10.5281/zenodo.23806339 (all versions).
