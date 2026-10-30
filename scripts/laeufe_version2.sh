#!/usr/bin/env bash
# Zusatzlaeufe fuer Version 2 (Review vom 22.10.2026), zwei Spuren parallel.
#   Spur 1: NMS-IoU abstimmen (32) -> YOLOv8s (300) -> YOLOv8n erweiterte Stufen
#   Spur 2: DINOv2-small Skala 1,5 (300) -> DINOv2-small erweiterte Stufen
# Erweiterte Stufen: gemessene GSD 8 / 11 / 15 / 20 mm/px = simulierte Hoehen
# 23,8 / 32,7 / 44,6 / 59,5 m (GSD = 1,714 * h / 5,1, wie in 20).
#   bash laeufe_version2.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf8
ERW="23.8 32.7 44.6 59.5"
SPERR=../ergebnisse/kv/kachelgruppen.json
mkdir -p ../ergebnisse/logs_v2

(
  "$PY" 32_nms_iou.py --device 0 > ../ergebnisse/logs_v2/32_nms_iou.log 2>&1
  bash laeufe_kv_pc.sh kv_bereinigt_v8s --sperrliste "$SPERR" --modell yolov8s.pt
  STUFEN="$ERW" bash laeufe_kv_pc.sh kv_bereinigt_erw --sperrliste "$SPERR"
  echo "Spur 1 fertig"
) > ../ergebnisse/logs_v2/spur1.log 2>&1 &

(
  "$PY" 25_ssl_kv.py --encoder dinov2_s --skala 1.5 --device cuda > ../ergebnisse/logs_v2/25_dinov2_x1.5.log 2>&1
  "$PY" 25_ssl_kv.py --encoder dinov2_s --stufen $ERW --suffix _erw --device cuda > ../ergebnisse/logs_v2/25_dinov2_erw.log 2>&1
  echo "Spur 2 fertig"
) > ../ergebnisse/logs_v2/spur2.log 2>&1 &

wait
echo "alle Zusatzlaeufe fertig"
