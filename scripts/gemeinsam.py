"""Gemeinsame Hilfsfunktionen fuer den Bluetenerkennungs-Prototyp.

Wichtige Annahme zur GSD-Berechnung
-----------------------------------
Die DJI M3T Weitwinkelkamera hat ca. 24 mm KB-Aequivalent. Daraus folgt eine
horizontale Bildwinkelbreite von 2*atan(36/(2*24)) = 73.7 Grad und damit eine
Bodenbreite von rund 1.50 * Aufnahmedistanz.

ACHTUNG - vor der Verwendung im Paper pruefen:
  * RelativeAltitude aus dem XMP ist die Hoehe ueber dem STARTPUNKT, nicht
    ueber dem Kronendach. Bei 2-3 m Baumhoehe ist die reale Aufnahmedistanz
    deutlich kleiner. Ueber --kronenhoehe korrigierbar.
  * Die GSD-Angaben in den Ordnernamen (0,1 / 0,15 / ... / 0,4) sind intern
    inkonsistent (0,3 bei 16,5 m vs. 0,4 bei 12 m). Vermutlich wurden sie in
    UgCS fuer 48 MP gerechnet, gespeichert sind aber 12 MP (4000x3000).
    Fuer alle Auswertungen die hier berechneten Werte verwenden, nicht die
    Ordnernamen.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

def ssd_wurzel() -> Path:
    """Wurzel der Drohnen-Rohdaten.

    Ueber die Umgebungsvariable BLUETEN_SSD setzbar, damit die Skripte ohne
    Aenderung auch unter Windows laufen, z. B.
        macOS    export BLUETEN_SSD="/Volumes/SSD_Drohne/Drohnendaten/Saison 2026/0428 Blütenerkennung Apfel"
        Windows  set BLUETEN_SSD=E:\\Drohnendaten\\Saison 2026\\0428 Blütenerkennung Apfel

    Ohne Umgebungsvariable wird die lokale Kopie unter
    rohbilder/0428 Blütenerkennung Apfel verwendet, falls vorhanden
    (Ablass-Hoehenleiter und Zauberwald vollstaendig, Am Weiler und Ragewitz
    nur als Stichprobe; angelegt mit scripts/ssd_sichern.ps1).
    """
    if "BLUETEN_SSD" in os.environ:
        return Path(os.environ["BLUETEN_SSD"])
    kopie = (Path(__file__).resolve().parent.parent / "rohbilder"
             / "0428 Blütenerkennung Apfel")
    if kopie.is_dir():
        return kopie
    return Path("/Volumes/SSD_Drohne/Drohnendaten/Saison 2026/0428 Blütenerkennung Apfel")


# Bodenbreite = FOV_FAKTOR * Aufnahmedistanz (siehe Modulkommentar)
FOV_FAKTOR = 1.50
SENSOR_BREITE_PX = 4000


def lies_xmp(pfad: Path) -> dict:
    """Liest die DJI-XMP-Felder aus dem JPEG-Header (ohne das Bild zu dekodieren)."""
    with open(pfad, "rb") as f:
        kopf = f.read(120_000)
    start, ende = kopf.find(b"<x:xmpmeta"), kopf.find(b"</x:xmpmeta>")
    if start < 0 or ende < 0:
        return {}
    text = kopf[start:ende].decode("utf8", "ignore")
    felder = {}
    for schluessel in ("RelativeAltitude", "AbsoluteAltitude", "GimbalPitchDegree",
                       "GimbalYawDegree", "FlightYawDegree"):
        treffer = re.search(schluessel + r'="([^"]*)"', text)
        if treffer:
            felder[schluessel] = float(treffer.group(1))
    return felder


def mm_pro_pixel(hoehe_ueber_start_m: float, kronenhoehe_m: float = 0.0) -> float:
    """Effektive Bodenaufloesung in mm/px."""
    distanz = max(0.5, hoehe_ueber_start_m - kronenhoehe_m)
    return 1000.0 * FOV_FAKTOR * distanz / SENSOR_BREITE_PX


def schaerfe(grau: np.ndarray) -> float:
    """Varianz des Laplace-Operators. Hoeher = schaerfer.

    Relevant, weil am 29.04.2026 starker Wind herrschte - Bewegungsunschaerfe
    ist eine Stoergroesse, die sauber herausgefiltert bzw. mitberichtet werden muss.
    """
    g = grau.astype(np.float32)
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def bluetenscore(rgb: np.ndarray, variante: str = "weiss_textur") -> dict:
    """Bewertet ein Fenster danach, wie wahrscheinlich es auf der Baumreihe liegt.

    Rueckgabe: dict mit 'score' und den Einzelmerkmalen.

    Hintergrund
    -----------
    Ein reiner Weiss-Schwellwert - die Grundlage der Verfahren von Tubau Comas
    (2019), Yuan (2023) und Piani (2021) - laesst sich in dieser Anlage nicht
    verwenden: abgetrockneter Sandboden hat dieselbe Helligkeits- und
    Saettigungssignatur wie Apfelbluete. Ein Weiss-Schwellwert landet
    zuverlaessig auf dem Boden statt auf der Bluete.

    Empirischer Vergleich (4 Bilder aus Flug 009, je bestes Fenster):
        veg * weiss                 2 von 4 auf der Reihe
        weiss * schatten            4 von 4
        weiss * textur              4 von 4   <- Voreinstellung
        weiss * schatten * textur   4 von 4
        veg * weiss * schatten      4 von 4

    Gewaehlt ist 'weiss_textur', weil der Schatten-Anteil stark vom Sonnenstand
    abhaengt. Fuer den spaeteren Vergleich von Fluegen zu verschiedenen
    Tageszeiten (Am Weiler, 15:24-16:48) darf die Kachelauswahl nicht mit dem
    Sonnenstand mitwandern - sonst ist der Tageszeiteffekt durch die
    Stichprobenziehung konfundiert. Textur ist in dieser Hinsicht neutraler.
    """
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hell, dunkel = a.max(2), a.min(2)
    grau = a.mean(2)

    m = {
        "vegetation": float((2 * g - r - b > 18).mean()),                      # Excess Green
        "weiss": float(((hell > 165) & ((hell - dunkel) < 45) & (g > 90)).mean()),
        "schatten": float((grau < 80).mean()),
        "textur": float(grau.std()) / 128.0,
    }

    formeln = {
        "weiss_textur": lambda d: d["weiss"] * d["textur"],
        "weiss_schatten": lambda d: d["weiss"] * d["schatten"],
        "veg_weiss": lambda d: d["vegetation"] * d["weiss"],
        "kombiniert": lambda d: d["weiss"] * d["schatten"] * d["textur"],
    }
    if variante not in formeln:
        raise ValueError(f"unbekannte Variante {variante!r}, "
                         f"waehle aus {sorted(formeln)}")

    m["score"] = formeln[variante](m)
    return m


def ls_dateiname(quelle: str) -> str:
    """Holt den Originaldateinamen aus einem Label-Studio-Datenpfad.

    Label Studio haengt beim Upload einen 8-stelligen Hex-Praefix an, etwa
    '/data/upload/1/1fa94995-DJI_2026...jpg'. Die Kachelnamen selbst enthalten
    keine Bindestriche, das Abtrennen ist also eindeutig.
    """
    name = Path(quelle.split("?d=")[-1].split("/")[-1]).name
    teile = name.split("-")
    if len(teile) > 1 and len(teile[0]) == 8 and teile[0].isalnum():
        name = "-".join(teile[1:])
    return name


def punkte_aus_export(export_json: Path, ausschluss=("unsicher",)) -> dict:
    """Liest einen Label-Studio-JSON-Export.

    Rueckgabe: Dateiname -> dict(punkte=[(x%, y%)...], unsicher=int)
    """
    import json
    daten = json.loads(Path(export_json).read_text(encoding="utf8"))
    ergebnis = {}
    for aufgabe in daten:
        name = ls_dateiname(aufgabe.get("data", {}).get("image", ""))
        punkte, unsicher = [], 0
        for annot in aufgabe.get("annotations", []):
            for r in annot.get("result", []):
                if r.get("type") != "keypointlabels":
                    continue
                wert = r["value"]
                if any(l in ausschluss for l in wert.get("keypointlabels", [])):
                    unsicher += 1
                    continue
                punkte.append((wert["x"], wert["y"]))
        ergebnis[name] = {"punkte": punkte, "unsicher": unsicher}
    return ergebnis


def unsichere_punkte(export_json: Path, label: str = "unsicher") -> dict:
    """Nur die als 'unsicher' markierten Punkte: Dateiname -> [(x%, y%), ...]."""
    import json
    daten = json.loads(Path(export_json).read_text(encoding="utf8"))
    aus = {}
    for aufgabe in daten:
        name = ls_dateiname(aufgabe.get("data", {}).get("image", ""))
        aus[name] = [(r["value"]["x"], r["value"]["y"])
                     for annot in aufgabe.get("annotations", []) for r in annot.get("result", [])
                     if r.get("type") == "keypointlabels"
                     and label in r["value"].get("keypointlabels", [])]
    return aus


def ignorierzone(det_xy: np.ndarray, sicher_px: np.ndarray, unsicher_px: np.ndarray,
                 radius: float = 12.0) -> np.ndarray:
    """Maske der Detektionen, die gewertet werden (Review 3.4, Version 2).

    Unsichere Punkte sind Ignorierzonen: Eine Detektion innerhalb `radius` eines
    unsicheren Punkts zaehlt weder als Treffer noch als Fehlalarm, ausser sie
    liegt auch innerhalb `radius` eines sicheren Referenzpunkts.
    """
    if len(det_xy) == 0 or len(unsicher_px) == 0:
        return np.ones(len(det_xy), bool)
    nah_u = (np.linalg.norm(det_xy[:, None, :2] - unsicher_px[None], axis=2) <= radius).any(1)
    if len(sicher_px):
        nah_s = (np.linalg.norm(det_xy[:, None, :2] - sicher_px[None], axis=2) <= radius).any(1)
    else:
        nah_s = np.zeros(len(det_xy), bool)
    return ~(nah_u & ~nah_s)


def _kastenmittel(a: np.ndarray, k: int) -> np.ndarray:
    """Mittelwert ueber ein k x k Fenster, per Integralbild.

    Ersetzt scipy.ndimage.uniform_filter. Bewusst ohne scipy: das in dieser
    Umgebung installierte scipy 1.12 ist gegen NumPy 1.x kompiliert und laeuft
    unter dem von ultralytics mitgebrachten NumPy 2.x nicht mehr.
    """
    k = max(1, int(k) | 1)
    r = k // 2
    pad = np.pad(a.astype(np.float64), r, mode="edge")
    s = np.pad(pad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = a.shape
    return ((s[2 * r + 1:h + 2 * r + 1, 2 * r + 1:w + 2 * r + 1]
             - s[0:h, 2 * r + 1:w + 2 * r + 1]
             - s[2 * r + 1:h + 2 * r + 1, 0:w]
             + s[0:h, 0:w]) / (k * k))


def _spitzen(profil: np.ndarray, mindestabstand: int,
             schwelle: float) -> np.ndarray:
    """Lokale Maxima mit Mindestabstand. Ersetzt scipy.signal.find_peaks."""
    innen = profil[1:-1]
    kand = np.where((innen > profil[:-2]) & (innen >= profil[2:]))[0] + 1
    kand = kand[profil[kand] >= schwelle]
    gewaehlt: list[int] = []
    for i in kand[np.argsort(-profil[kand])]:
        if all(abs(int(i) - j) >= mindestabstand for j in gewaehlt):
            gewaehlt.append(int(i))
    return np.array(sorted(gewaehlt), dtype=int)


def reihenkandidaten(bild, kachel_px: int, winkel_schritt: int = 3,
                     skala: int = 8) -> tuple[list[tuple[int, int]], float, int]:
    """Findet Fenstermittelpunkte, die auf den Baumreihen liegen.

    Fensterweise Bewertung (siehe bluetenscore) betrachtet jedes Fenster
    isoliert und ignoriert die staerkste vorhandene Struktur: Obstanlagen
    bestehen aus langen, parallelen, aequidistanten Reihen. Hier wird deshalb
    zuerst die Reihengeometrie geschaetzt und erst danach entlang der Reihen
    abgetastet.

    Verfahren
    ---------
    1. Texturkarte aufbauen (lokale Standardabweichung des Grauwerts). Laub und
       Schattenwurf der Reihe erzeugen hohe Textur, Grasgasse und abgetrockneter
       Boden sind glatt.
    2. Radon-artige Projektion: Fuer jeden Winkel wird die Texturkarte auf die
       Achse senkrecht zur Reihenrichtung projiziert. Liegt die Projektion
       parallel zu den Reihen, entsteht ein Profil mit scharfen Maxima; die
       Varianz des Profils ist dann maximal.
    3. Maxima im Profil des besten Winkels = Reihenmittellinien.
    4. Entlang jeder Mittellinie in Kachelabstaenden abtasten.

    Rueckgabe: (Liste von (x, y) Mittelpunkten in Originalpixeln,
                Reihenwinkel in Grad, Anzahl gefundener Reihen)
    """
    breite, hoehe = bild.size
    klein = np.asarray(bild.resize((breite // skala, hoehe // skala))
                       .convert("L")).astype(np.float32)
    h, w = klein.shape

    # lokale Standardabweichung ueber ein Fenster von ca. 0.5 m
    fenster = max(3, kachel_px // (skala * 4))
    mittel = _kastenmittel(klein, fenster)
    textur = np.sqrt(np.maximum(_kastenmittel(klein ** 2, fenster) - mittel ** 2, 0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X, Y = xx - w / 2.0, yy - h / 2.0
    gewicht = textur.ravel()

    bestes = (-1.0, 0.0, None, None)
    for grad in range(0, 180, winkel_schritt):
        rad = np.radians(grad)
        # t = Abstand senkrecht zur Reihenrichtung
        t = (-X * np.sin(rad) + Y * np.cos(rad)).ravel()
        idx = (t - t.min()).astype(np.int32)
        summe = np.bincount(idx, weights=gewicht)
        anzahl = np.bincount(idx)
        gueltig = anzahl > (0.25 * anzahl.max())
        if gueltig.sum() < 8:
            continue
        profil = np.zeros_like(summe)
        profil[gueltig] = summe[gueltig] / anzahl[gueltig]
        # nur den gut abgedeckten Mittelbereich bewerten
        score = float(profil[gueltig].var())
        if score > bestes[0]:
            bestes = (score, float(grad), profil, (t.min(), gueltig))

    _, grad, profil, (t_min, gueltig) = bestes
    rad = np.radians(grad)

    kachel_klein = max(4, kachel_px // skala)
    schwelle = float(np.median(profil[gueltig]) + 0.35 * profil[gueltig].std())
    spitzen = _spitzen(np.where(gueltig, profil, 0.0),
                       mindestabstand=max(3, int(kachel_klein * 0.6)),
                       schwelle=schwelle)

    u = np.array([np.cos(rad), np.sin(rad)])      # entlang der Reihe
    v = np.array([-np.sin(rad), np.cos(rad)])     # senkrecht dazu
    laenge = float(np.hypot(w, h))

    zentren = []
    for spitze in spitzen:
        t_k = spitze + t_min
        for s in np.arange(-laenge / 2, laenge / 2 + 1, kachel_klein):
            p = t_k * v + s * u
            x = int(round((p[0] + w / 2.0) * skala))
            y = int(round((p[1] + h / 2.0) * skala))
            if (kachel_px // 2 <= x <= breite - kachel_px // 2
                    and kachel_px // 2 <= y <= hoehe - kachel_px // 2):
                zentren.append((x, y))

    return zentren, grad, len(spitzen)


def lies_gps(pfad: Path) -> tuple[float, float] | None:
    """Liest (Breite, Laenge) in Dezimalgrad aus dem EXIF-GPS-Block."""
    from PIL import Image as _Image
    try:
        ex = _Image.open(pfad).getexif()
        gps = ex.get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def grad(werte, ref):
        g, m, s = (float(w) for w in werte)
        wert = g + m / 60.0 + s / 3600.0
        return -wert if ref in ("S", "W") else wert

    try:
        return (grad(gps[2], gps.get(1, "N")), grad(gps[4], gps.get(3, "E")))
    except Exception:
        return None


def meter_pro_grad(breite_grad: float) -> tuple[float, float]:
    """Naeherung fuer Meter je Grad Breite bzw. Laenge auf dieser Breite.

    Fuer Abstaende von wenigen hundert Metern voellig ausreichend; eine
    UTM-Projektion waere hier ueberdimensioniert.
    """
    import math
    rad = math.radians(breite_grad)
    m_breite = 111132.92 - 559.82 * math.cos(2 * rad) + 1.175 * math.cos(4 * rad)
    m_laenge = 111412.84 * math.cos(rad) - 93.5 * math.cos(3 * rad)
    return m_breite, m_laenge


def fussabdruck_m(hoehe_m: float, kronenhoehe_m: float = 0.0) -> tuple[float, float]:
    """Bodenabdeckung (Breite, Hoehe) eines Nadirbildes in Metern."""
    distanz = max(0.5, hoehe_m - kronenhoehe_m)
    breite = FOV_FAKTOR * distanz
    return breite, breite * 3.0 / 4.0        # 4000x3000 Sensor


def finde_bilder(ordner: Path) -> list[Path]:
    """Alle RGB-Bilder (_V) eines Flugordners, zeitlich sortiert."""
    bilder = sorted(p for p in ordner.iterdir()
                    if p.suffix.upper() == ".JPG" and "_V" in p.stem)
    return [p for p in bilder if Image.open(p).size[0] >= 2000]
