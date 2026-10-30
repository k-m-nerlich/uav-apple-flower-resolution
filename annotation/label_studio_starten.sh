#!/usr/bin/env bash
# Startet Label Studio lokal mit Zugriff auf den Kachelordner.
set -euo pipefail

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/km.nerlich/Desktop/Daten/2 Studium/8 Promotion/Claude Ansätze/bluetenerkennung/kacheln"

echo "Label Studio startet auf http://localhost:8080"
echo "Dokumentwurzel: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
label-studio start --data-dir "/Users/km.nerlich/Desktop/Daten/2 Studium/8 Promotion/Claude Ansätze/bluetenerkennung/annotation/ls_data"
