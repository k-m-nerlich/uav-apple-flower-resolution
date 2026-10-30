#!/usr/bin/env python3
"""Entpackt die Rohvorhersagen: ergebnisse/<variante>/vorhersagen.zip -> vorhersagen/.

Die Auswertungsskripte (21, 22, 23, 28, 29, 34, 35) lesen die entpackten
JSON-Dateien. Bereits entpackte Varianten werden uebersprungen.

    python entpacken.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent.parent


def main() -> None:
    for z in sorted((PROJEKT / "ergebnisse").glob("*/vorhersagen.zip")):
        ziel = z.parent / "vorhersagen"
        if ziel.exists() and any(ziel.iterdir()):
            print(f"{z.parent.name:<28} schon entpackt")
            continue
        with zipfile.ZipFile(z) as f:
            f.extractall(ziel)
            print(f"{z.parent.name:<28} {len(f.namelist()):4d} Dateien")


if __name__ == "__main__":
    main()
