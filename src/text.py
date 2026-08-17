"""Représentations textuelles.

Deux familles, mêmes conventions :

- **TF-IDF** : ajusté sur l'entraînement seulement. C'est la seule étape à état
  de la chaîne texte, donc le seul endroit où une fuite est possible.
- **Encodeur figé** : aucun état appris, la moyenne des jetons de
  `last_hidden_state`. Le modèle n'est jamais réglé finement : on compare des
  représentations, pas des capacités de modèles.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BATCH = 32
MAX_LEN = 256

ENCODEURS = {
    "BERT figé (2018)": "bert-base-uncased",
    "ModernBERT figé (2024)": "answerdotai/ModernBERT-base",
}


def vectoriseur() -> TfidfVectorizer:
    """Le vectoriseur du projet. Un seul endroit le définit."""
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )


def charger_encodeur(model_id: str, device: str | None = None):
    """Renvoie (tokeniseur, modèle, device). Le modèle est en mode évaluation."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id).to(device).eval()
    return tok, mdl, device


def encoder(textes: list[str], tok, mdl, device: str) -> np.ndarray:
    """Moyenne des jetons, masque d'attention pris en compte."""
    import torch

    sorties = []
    with torch.no_grad():
        for i in range(0, len(textes), BATCH):
            lot = tok(
                textes[i : i + BATCH],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            h = mdl(**lot).last_hidden_state
            masque = lot["attention_mask"].unsqueeze(-1).float()
            sorties.append(((h * masque).sum(1) / masque.sum(1)).cpu().numpy())
    return np.vstack(sorties)


def empreinte_mo(mdl) -> float:
    """Empreinte du modèle en float32, en mégaoctets."""
    return sum(p.numel() for p in mdl.parameters()) * 4 / 1e6
