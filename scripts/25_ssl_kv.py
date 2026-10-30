#!/usr/bin/env python3
"""Schritt 25: Selbstueberwacht vortrainierte Encoder (I-JEPA, DINOv2) in derselben
bereinigten Kreuzvalidierung wie YOLO (20/21) und Farbschwelle (23).

Frage
-----
Hilft eine Repraesentation, die OHNE Labels gelernt wurde, bei nur 19-24
annotierten Trainingskacheln? Gemessen gegen YOLOv8n (F1 0,489) und die
Farbschwelle (0,522) bei 1,71 mm/px, gleiche Falten, gleiche Sperrliste.

Aufbau ("Stufe 1": Encoder eingefroren)
---------------------------------------
  1. Encoder (z. B. I-JEPA ViT-H/14) einmal ueber alle 36 Kacheln jeder
     Aufloesungsstufe laufen lassen, Patch-Merkmale auf Platte cachen
     (merkmale/<encoder>_x<skala>/<stufe>.pt). Fuer das Training zusaetzlich
     gespiegelte Varianten (--varianten 4: original, links-rechts, oben-unten,
     beides), weil die Merkmale eingefroren sind und Bildaugmentierung
     sonst entfaellt.
  2. Kleiner Faltungskopf auf dem Patch-Raster (Ausgabe mit halber
     Patchweite, bei Patch 14 also alle 7 px): Punkt-Heatmap im CenterNet-Stil
     (Fokalverlust) plus Versatz innerhalb der Zelle. Nur der Kopf wird
     trainiert, je Falte und Seed neu.
  3. Bester Stand nach F1 auf der val-Falte (wie best.pt in 20), dann
     Rohvorhersagen ab conf 0,01 fuer val und test schreiben - im SELBEN
     Format wie 20. Auswertung daher unveraendert mit 21:

         python 21_kv_auswertung.py --kv kv_ssl_ijepa_h14_1k

Warum Patch-Raster und nicht Bildklassifikation: eine Bluete ist bei
1,71 mm/px etwa 14-24 px gross, also ungefaehr ein ViT-Patch. Die Kacheln
werden deshalb in voller Aufloesung eingespeist (1022 px bei Patch 14,
Raster 73 x 73), nicht auf 224 px verkleinert. --skala 1.5 vergroessert
zusaetzlich (Bluete ueber mehrere Patches, Rechenaufwand ~2,3x).

Seeds betreffen hier nur den Kopf (Initialisierung, Batchreihenfolge), der
Encoder ist fest. Die Seed-Streuung ist deshalb nicht direkt mit der von YOLO
vergleichbar, die Faltenstreuung schon.

Encoder (--encoder, Hugging-Face-Kennung oder Kurzname)
    ijepa_h14_1k   facebook/ijepa_vith14_1k    I-JEPA ViT-H/14, ImageNet-1k
    ijepa_h14_22k  facebook/ijepa_vith14_22k   I-JEPA ViT-H/14, ImageNet-22k
    ijepa_g16_22k  facebook/ijepa_vitg16_22k   I-JEPA ViT-g/16 (gross, langsam)
    dinov2_s/b/l   facebook/dinov2-small/-base/-large   Kontrolle: anderes
                   selbstueberwachtes Verfahren, gleiche Bauart (ViT/14)

Beispiele
    python 25_ssl_kv.py --encoder dinov2_s --stufen 5.1 --seeds 0 --nur-falten 0 --epochen 20   (Probelauf)
    python 25_ssl_kv.py --encoder ijepa_h14_1k --device cuda
    python 25_ssl_kv.py --encoder ijepa_h14_1k --max-train 12      (Lernkurve)

Wiederaufnehmbar: vorhandene Vorhersagedateien und Merkmal-Caches werden
uebersprungen.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from gemeinsam import mm_pro_pixel, punkte_aus_export

PROJEKT = Path(__file__).resolve().parent.parent
ERG = PROJEKT / "ergebnisse"
STUFEN_M = [5.1, 8.8, 10.7, 14.8, 16.7]          # wie 20
NATIVE_GSD_GEMESSEN = 1.714                       # wie 20
ENCODER = {
    "ijepa_h14_1k": "facebook/ijepa_vith14_1k",
    "ijepa_h14_22k": "facebook/ijepa_vith14_22k",
    "ijepa_g16_22k": "facebook/ijepa_vitg16_22k",
    "dinov2_s": "facebook/dinov2-small",
    "dinov2_b": "facebook/dinov2-base",
    "dinov2_l": "facebook/dinov2-large",
}
MITTEL = (0.485, 0.456, 0.406)                    # ImageNet, fuer beide Familien
STD = (0.229, 0.224, 0.225)
VARIANTEN = [(False, False), (True, False), (False, True), (True, True)]  # (lr, ou)
CONF_VAL = np.round(np.arange(0.05, 0.951, 0.05), 2)


def marke_von(hoehe: float) -> str:
    return f"h{round(hoehe * 10):03d}"


def degradieren(bild: Image.Image, faktor: float, kachel_px: int) -> Image.Image:
    """Wie 20.baue_stufe: LANCZOS herunter, BICUBIC zurueck."""
    if faktor < 0.999:
        klein = max(32, int(round(kachel_px * faktor)))
        bild = (bild.resize((klein, klein), Image.LANCZOS)
                    .resize((kachel_px, kachel_px), Image.BICUBIC))
    return bild


def zuordnen(wahrheit: np.ndarray, vorhersage: np.ndarray, radius: float):
    """Greedy-Zuordnung nach Distanz, wie 21. Rueckgabe (TP, FP, FN)."""
    if len(wahrheit) == 0:
        return 0, len(vorhersage), 0
    if len(vorhersage) == 0:
        return 0, 0, len(wahrheit)
    d = np.linalg.norm(wahrheit[:, None, :] - vorhersage[None, :, :], axis=2)
    fw = np.ones(len(wahrheit), bool)
    fv = np.ones(len(vorhersage), bool)
    tp = 0
    for iw, iv in sorted(np.argwhere(d <= radius).tolist(),
                         key=lambda p: d[p[0], p[1]]):
        if fw[iw] and fv[iv]:
            fw[iw] = fv[iv] = False
            tp += 1
    return tp, int(fv.sum()), int(fw.sum())


def f1(tp, fp, fn) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


# ---------------------------------------------------------------- Encoder ---

def lade_encoder(name: str, device: str):
    import torch
    from transformers import AutoModel
    kennung = ENCODER.get(name, name)
    modell = AutoModel.from_pretrained(kennung).eval().to(device)
    for par in modell.parameters():
        par.requires_grad_(False)
    patch = modell.config.patch_size
    patch = patch[0] if isinstance(patch, (list, tuple)) else int(patch)
    interp = "interpolate_pos_encoding" in inspect.signature(modell.forward).parameters
    return modell, patch, interp, kennung


def merkmale_einer_kachel(modell, patch: int, interp: bool, bild: Image.Image,
                          groesse: int, device: str, schicht: int):
    """Patch-Merkmale [D, g, g] als float16 auf der CPU."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.asarray(bild, dtype=np.uint8).copy()).permute(2, 0, 1)
    x = x.float()[None] / 255.0
    if x.shape[-1] != groesse:
        x = F.interpolate(x, size=(groesse, groesse), mode="bicubic",
                          align_corners=False).clamp(0, 1)
    x = (x - torch.tensor(MITTEL)[None, :, None, None]) / torch.tensor(STD)[None, :, None, None]
    x = x.to(device)
    kw = {"pixel_values": x, "output_hidden_states": schicht != -1}
    if interp:
        kw["interpolate_pos_encoding"] = True
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                         enabled=device.startswith("cuda")):
        aus = modell(**kw)
    h = aus.last_hidden_state if schicht == -1 else aus.hidden_states[schicht]
    g = groesse // patch
    tok = h[0, -g * g:, :].float()                 # CLS/Register vorne abschneiden
    return tok.T.reshape(-1, g, g).half().cpu()


def baue_cache(args, kacheln: list[str], original: dict, faktor: float,
               ziel: Path, device: str, enc) -> dict:
    import torch
    if ziel.exists():
        c = torch.load(ziel, map_location="cpu")
        if c["namen"] == kacheln and c["varianten"] >= args.varianten:
            return c
    modell, patch, interp, kennung = enc()
    groesse = int(round(args.kachel_px * args.skala / patch)) * patch
    t0 = time.time()
    alle = []
    for i, n in enumerate(kacheln):
        b = degradieren(original[n], faktor, args.kachel_px)
        je = []
        for lr, ou in VARIANTEN[:args.varianten]:
            v = b
            if lr:
                v = v.transpose(Image.FLIP_LEFT_RIGHT)
            if ou:
                v = v.transpose(Image.FLIP_TOP_BOTTOM)
            je.append(merkmale_einer_kachel(modell, patch, interp, v, groesse,
                                            device, args.schicht))
        alle.append(torch.stack(je))
        if i == 0:
            print(f"  Merkmale {tuple(alle[0].shape[1:])} je Kachel und Variante, "
                  f"erste Kachel {time.time() - t0:.1f} s", flush=True)
    c = {"namen": kacheln, "varianten": args.varianten, "patch": patch,
         "groesse": groesse, "kennung": kennung, "schicht": args.schicht,
         "merkmale": torch.stack(alle)}          # [N, V, D, g, g] float16
    ziel.parent.mkdir(parents=True, exist_ok=True)
    torch.save(c, ziel)
    print(f"  Cache {ziel.name}: {c['merkmale'].numel() * 2 / 1e9:.2f} GB, "
          f"{time.time() - t0:.0f} s", flush=True)
    return c


# ------------------------------------------------------------------- Kopf ---

def baue_kopf(d: int):
    import torch.nn as nn

    class Kopf(nn.Module):
        def __init__(self):
            super().__init__()
            self.ein = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(d, 256, 1), nn.GroupNorm(32, 256), nn.GELU(),
                nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.GELU())
            self.hoch = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
                nn.Conv2d(128, 3, 1))
            self.hoch[-1].bias.data[0] = -2.19     # Startwahrscheinlichkeit ~0,1

        def forward(self, x):
            return self.hoch(self.ein(x))          # [B, 3, 2g, 2g]: Heat, dx, dy

    return Kopf()


def ziele(punkte: np.ndarray, lr: bool, ou: bool, K: int, groesse: int,
          G: int, sigma_px: float):
    """Heatmap [G, G], Versatz [2, G, G], Maske [G, G] fuer eine Kachel/Variante."""
    heat = np.zeros((G, G), np.float32)
    vers = np.zeros((2, G, G), np.float32)
    maske = np.zeros((G, G), np.float32)
    if len(punkte) == 0:
        return heat, vers, maske
    p = punkte.copy()
    if lr:
        p[:, 0] = K - p[:, 0]
    if ou:
        p[:, 1] = K - p[:, 1]
    schritt = groesse / G                          # Eingabepixel je Ausgabezelle
    u = p * (groesse / K) / schritt                # kontinuierliche Rasterkoordinate
    # sigma_px: Zahl (fest) oder ein Wert je Punkt (Groesse je Bluete, 31)
    s = np.broadcast_to(np.asarray(sigma_px, float), (len(p),)) * (groesse / K) / schritt
    mitte = np.arange(G) + 0.5
    dx = (mitte[None, :] - u[:, 0, None]) ** 2     # [n, G]
    dy = (mitte[None, :] - u[:, 1, None]) ** 2
    heat = np.exp(-(dy[:, :, None] + dx[:, None, :])
                  / (2 * s[:, None, None] ** 2)).max(0).astype(np.float32)
    c = np.clip(np.floor(u).astype(int), 0, G - 1)
    heat[c[:, 1], c[:, 0]] = 1.0
    vers[0, c[:, 1], c[:, 0]] = np.clip(u[:, 0] - c[:, 0], 0, 1)
    vers[1, c[:, 1], c[:, 0]] = np.clip(u[:, 1] - c[:, 1], 0, 1)
    maske[c[:, 1], c[:, 0]] = 1.0
    return heat, vers, maske


def fokalverlust(logit, ziel):
    import torch
    p = logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = ziel.eq(1).float()
    l_pos = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    l_neg = -(torch.log(1 - p) * p ** 2 * (1 - ziel) ** 4 * (1 - pos)).sum()
    return (l_pos + l_neg) / pos.sum().clamp(min=1)


def dekodieren(aus, K: int, groesse: int, conf_min: float, max_det: int) -> np.ndarray:
    """Lokale Maxima der Heatmap -> [[x, y, conf]] in Kachelpixeln."""
    import torch.nn.functional as F
    heat = aus[0].sigmoid()
    spitze = (heat == F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]) & (heat >= conf_min)
    ys, xs = spitze.nonzero(as_tuple=True)
    if len(ys) == 0:
        return np.zeros((0, 3))
    G = heat.shape[-1]
    schritt = groesse / G
    off = aus[1:3, ys, xs].sigmoid()
    x = (xs.float() + off[0]) * schritt * K / groesse
    y = (ys.float() + off[1]) * schritt * K / groesse
    a = np.stack([x.cpu().numpy(), y.cpu().numpy(), heat[ys, xs].float().cpu().numpy()], 1)
    return a[np.argsort(-a[:, 2])][:max_det]


def vorhersagen(kopf, M, idx: list[int], K: int, groesse: int, max_det: int):
    import torch
    kopf.eval()
    aus = {}
    with torch.no_grad():
        for i in idx:
            o = kopf(M[i, 0][None].float())[0]
            aus[i] = dekodieren(o, K, groesse, 0.01, max_det)
    kopf.train()
    return aus


def bestes_val_f1(vorh: dict, ref: dict, radius: float) -> float:
    beste = 0.0
    for c in CONF_VAL:
        s = np.zeros(3)
        for i, a in vorh.items():
            s += zuordnen(ref[i], a[a[:, 2] >= c, :2], radius)
        beste = max(beste, f1(*s))
    return beste


def trainiere_lauf(args, M, Z, ref_idx, teile_idx: dict, seed: int, device: str,
                   K: int, groesse: int):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kopf = baue_kopf(M.shape[2]).to(device)
    opt = torch.optim.AdamW(kopf.parameters(), lr=args.lr, weight_decay=args.wd)
    proben = [(i, v) for i in teile_idx["train"] for v in range(args.varianten)]
    schritte_je_epoche = int(np.ceil(len(proben) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochen * schritte_je_epoche, pct_start=0.1)
    heat, vers, maske = Z
    bester = (-1.0, 0, None)
    for ep in range(args.epochen):
        reihen = rng.permutation(len(proben))
        for b in range(schritte_je_epoche):
            wahl = [proben[j] for j in reihen[b * args.batch:(b + 1) * args.batch]]
            ii = torch.tensor([i for i, _ in wahl], device=device)
            vv = torch.tensor([v for _, v in wahl], device=device)
            x = M[ii, vv].float()
            out = kopf(x)
            h, o, m = heat[ii, vv], vers[ii, vv], maske[ii, vv]
            verlust = fokalverlust(out[:, 0], h)
            verlust = verlust + (((out[:, 1:3].sigmoid() - o).abs() * m[:, None]).sum()
                                 / m.sum().clamp(min=1))
            opt.zero_grad(set_to_none=True)
            verlust.backward()
            opt.step()
            sched.step()
        if (ep + 1) % args.eval_alle == 0 or ep + 1 == args.epochen:
            v = vorhersagen(kopf, M, teile_idx["val"], K, groesse, args.max_det)
            f = bestes_val_f1(v, ref_idx, args.radius_px)
            if f > bester[0]:
                bester = (f, ep + 1, copy.deepcopy(kopf.state_dict()))
    kopf.load_state_dict(bester[2])
    return kopf, bester[0], bester[1]


# ------------------------------------------------------------------- Ablauf --

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="ijepa_h14_1k",
                   help="Kurzname (siehe oben) oder Hugging-Face-Kennung")
    p.add_argument("--export", type=Path, default=PROJEKT / "annotation" / "export.json")
    p.add_argument("--kv-vorlage", default="kv_bereinigt",
                   help="Falten und Sperrliste von hier uebernehmen")
    p.add_argument("--stufen", type=float, nargs="+", default=STUFEN_M)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--nur-falten", type=int, nargs="+", default=None)
    p.add_argument("--max-train", type=int, default=None,
                   help="nur so viele Trainingskacheln je Falte (Lernkurve)")
    p.add_argument("--skala", type=float, default=1.0,
                   help="Eingabe = 1024 px x skala, auf Vielfaches der Patchweite")
    p.add_argument("--schicht", type=int, default=-1,
                   help="-1 = letzte Schicht, sonst Index in hidden_states")
    p.add_argument("--varianten", type=int, default=4, choices=[1, 2, 4])
    p.add_argument("--epochen", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--eval-alle", type=int, default=10)
    p.add_argument("--sigma-px", type=float, default=5.0,
                   help="Breite der Ziel-Heatmap in Kachelpixeln")
    p.add_argument("--radius-px", type=float, default=12.0)
    p.add_argument("--kachel-px", type=int, default=1024)
    p.add_argument("--max-det", type=int, default=5000)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (Voreinstellung: automatisch)")
    p.add_argument("--nur-merkmale", action="store_true",
                   help="nur die Merkmal-Caches anlegen, nicht trainieren")
    p.add_argument("--suffix", default="")
    p.add_argument("--groessen", type=Path, default=None,
                   help="annotation/punktgroessen.json (31): Heatmap-Breite je Bluete "
                        "= sigma_px * d / median(d), begrenzt auf [sigma/2, 2*sigma]; "
                        "Ausgabe dann mit Suffix _gr")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    kurz = args.encoder if args.encoder in ENCODER else args.encoder.split("/")[-1]
    tag = f"{kurz}" + (f"_x{args.skala:g}" if args.skala != 1 else "")
    suffix = args.suffix or ((f"_n{args.max_train}" if args.max_train else "")
                             + ("_gr" if args.groessen else ""))
    kv_name = f"kv_ssl_{tag}{suffix}"
    kv = ERG / kv_name
    (kv / "vorhersagen").mkdir(parents=True, exist_ok=True)

    vorlage = json.loads((ERG / args.kv_vorlage / "falten.json").read_text(encoding="utf8"))
    falten, teilungen = vorlage["falten"], copy.deepcopy(vorlage["teilungen"])
    if args.max_train:
        for k, t in enumerate(teilungen):
            if len(t["train"]) > args.max_train:
                t["train"] = sorted(random.Random(1000 + k).sample(t["train"], args.max_train))
    (kv / "falten.json").write_text(json.dumps(
        {**vorlage, "teilungen": teilungen, "encoder": args.encoder,
         "max_train": args.max_train}, indent=1), encoding="utf8")

    kacheln = sorted(x for f in falten for x in f)
    ordner = [d for d in sorted((PROJEKT / "kacheln").glob("*")) if d.is_dir()][0]
    roh = punkte_aus_export(args.export)
    K = args.kachel_px
    ref = [np.array(roh[n]["punkte"], float).reshape(-1, 2) / 100 * K for n in kacheln]
    original = {n: Image.open(ordner / n).convert("RGB") for n in kacheln}
    k_index = {n: i for i, n in enumerate(kacheln)}
    if args.groessen:
        g = json.loads(args.groessen.read_text(encoding="utf8"))["groessen"]
        med = float(np.median([v for n in kacheln for v in g[n]]))
        sigma_je = []
        for i, n in enumerate(kacheln):
            if len(g[n]) != len(ref[i]):
                raise SystemExit(f"Groessendatei passt nicht zu {n}")
            sigma_je.append(np.clip(args.sigma_px * np.array(g[n]) / med,
                                    args.sigma_px / 2, args.sigma_px * 2))
    else:
        sigma_je = [args.sigma_px] * len(kacheln)

    print(f"Encoder {args.encoder} -> {kv_name}, Geraet {device}")
    for k, t in enumerate(teilungen):
        print(f"  Falte {k}: train {len(t['train'])}, val {len(t['val'])}, test {len(t['test'])}")

    geladen = {}

    def enc():
        if "m" not in geladen:
            print(f"  lade {ENCODER.get(args.encoder, args.encoder)} ...", flush=True)
            geladen["m"] = lade_encoder(args.encoder, device)
        return geladen["m"]

    basis = mm_pro_pixel(STUFEN_M[0])
    for hoehe in args.stufen:
        marke = marke_von(hoehe)
        faktor = basis / mm_pro_pixel(hoehe)
        print(f"\n=== {marke} ({NATIVE_GSD_GEMESSEN / faktor:.2f} mm/px) ===", flush=True)
        cache = baue_cache(args, kacheln, original, faktor,
                           PROJEKT / "merkmale" / tag / f"{marke}.pt", device, enc)
        if args.nur_merkmale:
            continue
        M = cache["merkmale"][:, :args.varianten].to(device)   # [N, V, D, g, g] fp16
        groesse = cache["groesse"]
        G = 2 * M.shape[-1]
        Zh, Zv, Zm = [], [], []
        for i in range(len(kacheln)):
            hh, vv, mm = zip(*[ziele(ref[i], lr, ou, K, groesse, G, sigma_je[i])
                               for lr, ou in VARIANTEN[:args.varianten]])
            Zh.append(np.stack(hh)); Zv.append(np.stack(vv)); Zm.append(np.stack(mm))
        Z = tuple(torch.from_numpy(np.stack(a)).to(device) for a in (Zh, Zv, Zm))
        ref_idx = {i: ref[i] for i in range(len(kacheln))}

        for seed in args.seeds:
            for k in (args.nur_falten or range(len(falten))):
                lauf = f"{marke}_f{k}_s{seed}"
                ziel = kv / "vorhersagen" / f"{lauf}.json"
                if ziel.exists():
                    continue
                t0 = time.time()
                teile_idx = {t: [k_index[n] for n in teilungen[k][t]]
                             for t in ("train", "val", "test")}
                kopf, val_f1, ep = trainiere_lauf(args, M, Z, ref_idx, teile_idx,
                                                  seed, device, K, groesse)
                als_liste = lambda d_: {kacheln[i]: [[round(float(x), 1), round(float(y), 1),
                                                      round(float(c), 4)] for x, y, c in a]
                                        for i, a in d_.items()}
                daten = {
                    "hoehe_m": hoehe, "faktor": round(faktor, 4),
                    "gsd_nominell": round(mm_pro_pixel(hoehe), 3),
                    "gsd_gemessen": round(NATIVE_GSD_GEMESSEN / faktor, 3),
                    "falte": k, "seed": seed, "max_det": args.max_det,
                    "encoder": cache["kennung"], "skala": args.skala,
                    "schicht": args.schicht, "beste_epoche": ep,
                    "val_f1_bestschwelle": round(val_f1, 4),
                    "test": als_liste(vorhersagen(kopf, M, teile_idx["test"], K,
                                                  groesse, args.max_det)),
                    "val": als_liste(vorhersagen(kopf, M, teile_idx["val"], K,
                                                 groesse, args.max_det)),
                }
                ziel.write_text(json.dumps(daten), encoding="utf8")
                print(f"### {lauf}: val-F1 {val_f1:.3f} (Epoche {ep}), "
                      f"{time.time() - t0:.0f} s", flush=True)
        del M, Z
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nAuswertung: python 21_kv_auswertung.py --kv {kv_name}")


if __name__ == "__main__":
    main()
