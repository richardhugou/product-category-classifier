"""Coût d'inférence de la solution retenue, mesuré poste par poste.

Le rapport a besoin de chiffres, pas d'ordres de grandeur supposés. Trois postes
sont mesurés séparément sur cette machine, puis extrapolés à des volumes de
catalogue :

    extraction DINOv2      un passage avant par photographie, poste dominant
    vectorisation TF-IDF   transformation du texte, sans état à recalculer
    tête de classification prédiction des sept probabilités

    python scripts/cout_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion import concatener, normaliser  # noqa: E402
from src.pipeline import TEXT_COL, load, split  # noqa: E402
from src.pretraitement import chemin_image  # noqa: E402

ARTEFACT = ROOT / "models" / "fusion_tfidf_dinov2.joblib"
SORTIE = ROOT / "reports" / "cout_inference.json"
N_IMAGES = 16  # échantillon suffisant : l'écart-type du passage avant est faible
VOLUMES = (1_000, 10_000, 100_000)


def _mesurer_dinov2(chemins: list[Path]) -> tuple[float, str]:
    """Millisecondes par photographie, cache contourné."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from src.images import MODEL_ID

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    modele = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    images = [Image.open(c).convert("RGB") for c in chemins]
    with torch.no_grad():  # un passage à blanc : on ne mesure pas l'allocation initiale
        entrees = proc(images=images[:2], return_tensors="pt").to(device)
        modele(**entrees)

    debut = time.perf_counter()
    with torch.no_grad():
        for img in images:
            entrees = proc(images=img, return_tensors="pt").to(device)
            modele(**entrees)
    ecoule = time.perf_counter() - debut
    return ecoule / len(images) * 1000, device


def main() -> int:
    if not ARTEFACT.exists():
        print(f"Artefact absent : {ARTEFACT.name}. Lancer scripts/exporter_fusion.py.")
        return 1

    paquet = joblib.load(ARTEFACT)
    df = load()
    _, _, test = split(df)

    chemins = [chemin_image(u) for u in test["uniq_id"].head(N_IMAGES)]
    ms_image, device = _mesurer_dinov2(chemins)

    textes = test[TEXT_COL].tolist()
    debut = time.perf_counter()
    X_txt = paquet["vectoriseur"].transform(textes)
    ms_texte = (time.perf_counter() - debut) / len(textes) * 1000

    faux_dino = np.zeros((len(textes), 768 * 2), dtype=np.float32)
    X = concatener(X_txt, normaliser(faux_dino))
    debut = time.perf_counter()
    paquet["tete"].predict_proba(X)
    ms_tete = (time.perf_counter() - debut) / len(textes) * 1000

    total = ms_image + ms_texte + ms_tete
    mesures = {
        "machine": f"local, {device}",
        "postes_ms_par_article": {
            "extraction DINOv2": round(ms_image, 2),
            "vectorisation TF-IDF": round(ms_texte, 3),
            "tête de classification": round(ms_tete, 3),
            "total": round(total, 2),
        },
        "part_extraction_image": round(ms_image / total, 4),
        "artefact_Mo": round(ARTEFACT.stat().st_size / 1e6, 1),
        "heures_de_calcul_par_volume": {
            f"{v:_}".replace("_", " ") + " articles": round(v * total / 1000 / 3600, 2)
            for v in VOLUMES
        },
    }
    SORTIE.write_text(json.dumps(mesures, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Machine : {mesures['machine']} · échantillon de {N_IMAGES} photographies\n")
    for poste, v in mesures["postes_ms_par_article"].items():
        print(f"  {poste:<24s} {v:>8.2f} ms/article")
    print(f"\n  part de l'extraction image : {mesures['part_extraction_image']:.1%}")
    print(f"  artefact sérialisé : {mesures['artefact_Mo']} Mo")
    print()
    for volume, heures in mesures["heures_de_calcul_par_volume"].items():
        print(f"  {volume:>18s} : {heures:>6.2f} h de calcul")
    print(f"\n→ {SORTIE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
