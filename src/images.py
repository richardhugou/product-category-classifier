"""Caractéristiques visuelles : encodeur figé, mises en cache sur disque.

Complément aux caractéristiques VGG16 de l'étude : DINOv2 est entraîné en
auto-supervision précisément pour produire de bonnes représentations **figées**,
ce qui n'est pas le cas d'un modèle de langage masqué dont on moyennerait les
jetons. Hors du périmètre de la mission, qui impose un réseau convolutif.

Représentation retenue : jeton de classe concaténé à la moyenne des jetons de
patch : le protocole d'évaluation linéaire de DINOv2.

    python -m src.images        # précalcule et met en cache
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "Flipkart" / "Images"
CACHE = ROOT / "data" / "features"
MODEL_ID = "facebook/dinov2-base"
BATCH = 16


def image_path(uniq_id: str) -> Path:
    return IMAGES / f"{uniq_id}.jpg"


def _load_encoder(device: str):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    return proc, model


def encode(uniq_ids, proc, model, device: str) -> np.ndarray:
    """Encode une liste d'articles. Renvoie (n, 1536)."""
    import torch
    from PIL import Image

    sorties = []
    with torch.no_grad():
        for i in range(0, len(uniq_ids), BATCH):
            lot = uniq_ids[i : i + BATCH]
            images = [Image.open(image_path(u)).convert("RGB") for u in lot]
            entrees = proc(images=images, return_tensors="pt").to(device)
            h = model(**entrees).last_hidden_state  # (b, 1 + patches, d)
            cls, patches = h[:, 0], h[:, 1:].mean(dim=1)
            sorties.append(torch.cat([cls, patches], dim=-1).cpu().numpy())
    return np.vstack(sorties)


def features(uniq_ids, device: str | None = None) -> tuple[np.ndarray, float]:
    """Caractéristiques visuelles, depuis le cache si possible.

    Renvoie (matrice, secondes d'encodage). Le temps vaut 0 si le cache a servi.

    Le cache est lu avant tout import de `torch` : relire des caractéristiques
    déjà calculées ne doit pas exiger un cadriciel d'apprentissage profond.
    C'est ce qui permet à `optimize.py` de tourner sur le socle seul.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_x = CACHE / "dinov2_base.npy"
    cache_ids = CACHE / "dinov2_base_ids.npy"

    ids = np.asarray(list(uniq_ids))
    if cache_x.exists() and cache_ids.exists():
        connus = np.load(cache_ids, allow_pickle=True)
        if len(connus) == len(ids) and (connus == ids).all():
            return np.load(cache_x), 0.0

    import torch

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    proc, model = _load_encoder(device)
    t0 = time.perf_counter()
    X = encode(list(ids), proc, model, device)
    secondes = time.perf_counter() - t0

    np.save(cache_x, X)
    np.save(cache_ids, ids)
    return X, secondes


def empreinte_encodeur_mo() -> float:
    """Empreinte de l'encodeur en mémoire, en float32."""

    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID)
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6


if __name__ == "__main__":
    from src.pipeline import load

    df = load()
    X, s = features(df["uniq_id"])
    print(f"{X.shape[0]} images encodées en {X.shape[1]} dimensions")
    print(f"temps d'encodage : {s:.1f} s" if s else "servi depuis le cache")
    print(f"cache : {CACHE}")
