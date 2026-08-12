"""Le vectoriseur est la seule étape à état de la chaîne texte."""

from __future__ import annotations

from src.text import vectoriseur


def test_ajuste_sur_entrainement_seulement():
    """Un terme vu uniquement dans le test ne doit pas entrer au vocabulaire.

    C'est la formulation opérationnelle de « pas de fuite » : le vocabulaire est
    une statistique de l'entraînement, il ignore le test par construction.
    """
    entrainement = ["ceramic vase decorative", "ceramic mug kitchen", "vase decorative table"]
    test = ["zirconium spectrometer anomaly"]

    vec = vectoriseur()
    vec.fit(entrainement)
    vocabulaire = set(vec.get_feature_names_out())

    assert not {"zirconium", "spectrometer", "anomaly"} & vocabulaire
    X = vec.transform(test)
    assert X.shape == (1, len(vocabulaire))
    assert X.nnz == 0  # aucun terme connu : le vecteur est nul, pas une erreur


def test_min_df_ecarte_les_termes_uniques():
    """min_df=2 : un terme vu une seule fois n'est pas retenu."""
    vec = vectoriseur()
    vec.fit(["vase ceramic", "vase ceramic", "singleton"])
    assert "singleton" not in set(vec.get_feature_names_out())


def test_sortie_normalisee_l2():
    """Le TF-IDF de scikit-learn normalise en L2 : les deux blocs de la fusion
    arrivent donc à la même échelle."""
    import numpy as np

    vec = vectoriseur()
    X = vec.fit_transform(["ceramic vase table", "ceramic mug table", "table vase ceramic"])
    normes = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(normes, 1.0)
