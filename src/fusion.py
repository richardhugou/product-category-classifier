"""Fusion des représentations texte et image.

La normalisation L2 est faite **ligne par ligne**. C'est le point important :
chaque vecteur est divisé par sa propre norme, donc la transformation n'a aucun
paramètre appris entre échantillons. Elle ne peut pas fuir d'un pli à l'autre,
et elle met les deux blocs à la même échelle avant concaténation : le TF-IDF de
scikit-learn étant déjà normalisé en L2 par défaut.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.preprocessing import normalize


def normaliser(X: np.ndarray) -> np.ndarray:
    """Normalisation L2 ligne par ligne. Sans état, donc sans fuite possible."""
    return normalize(np.atleast_2d(X))


def concatener(X_texte, X_image: np.ndarray):
    """Concatène le bloc texte (creux ou dense) et le bloc image normalisé."""
    bloc_image = csr_matrix(normaliser(X_image))
    if not issparse(X_texte):
        X_texte = csr_matrix(X_texte)
    return hstack([X_texte, bloc_image]).tocsr()
