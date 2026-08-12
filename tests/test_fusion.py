"""La fusion ne doit rien apprendre entre échantillons."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.fusion import concatener, normaliser


def test_normalisation_est_ligne_par_ligne():
    X = np.array([[3.0, 4.0], [0.0, 5.0]])
    N = normaliser(X)
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0)
    assert np.allclose(N[0], [0.6, 0.8])


def test_normalisation_sans_etat():
    """Normaliser un sous-ensemble donne les mêmes lignes que normaliser le tout.

    C'est la démonstration qu'aucune information ne circule entre les plis :
    la transformation ne dépend que de la ligne qu'elle traite.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    complet = normaliser(X)
    partiel = normaliser(X[5:12])
    assert np.allclose(complet[5:12], partiel)


def test_concatenation_dimensions():
    texte = csr_matrix(np.eye(4, 6))
    image = np.arange(4 * 3, dtype=float).reshape(4, 3) + 1
    F = concatener(texte, image)
    assert F.shape == (4, 9)


def test_concatenation_preserve_le_bloc_texte():
    texte = csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    image = np.array([[3.0, 4.0], [5.0, 12.0]])
    F = concatener(texte, image).toarray()
    assert np.allclose(F[:, :2], texte.toarray())
    assert np.allclose(F[0, 2:], [0.6, 0.8])
