#!/usr/bin/env bash
# Kreuzvalidierung (20_kreuzvalidierung.py): 5 Stufen x 6 Falten x 10 Seeds.
# Je Stufe ein Prozess, alle parallel auf der GPU. Wiederaufnehmbar.
#
#   bash laeufe_kv_pc.sh                       (erste KV, ergebnisse/kv)
#   bash laeufe_kv_pc.sh kv_bereinigt --sperrliste ../ergebnisse/kv/kachelgruppen.json
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
export PYTHONIOENCODING=utf8
AUSGABE=${1:-kv}; shift || true
mkdir -p "../ergebnisse/$AUSGABE/logs"

for h in ${STUFEN:-5.1 8.8 10.7 14.8 16.7}; do   # STUFEN="23.8 32.7" ueberschreibt
  "$PY" 20_kreuzvalidierung.py --stufen "$h" --seeds 0 1 2 3 4 5 6 7 8 9 \
      --device 0 --ausgabe "$AUSGABE" "$@" > "../ergebnisse/$AUSGABE/logs/stufe_${h}.log" 2>&1 &
  sleep 20   # Datensatzaufbau und Modellladen nicht gleichzeitig
done
wait
echo "alle Stufen fertig"
