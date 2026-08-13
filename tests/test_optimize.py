"""L'exploration des hyperparamètres doit être économe et honnête.

Deux propriétés sont vérifiées ici. La première évite de payer plusieurs fois
la même mesure : un plafond de vocabulaire qui ne mord pas produit exactement
la même représentation, et la grille doit s'en apercevoir. La seconde protège
la lecture des résultats : les effets marginaux sont ce sur quoi le rapport
s'appuie pour dire ce qui compte, alors autant vérifier qu'ils comptent juste.
"""

from __future__ import annotations

import pandas as pd

from optimize import _effets_marginaux, _grille_tfidf_utile

TEXTES = [
    "cotton printed king sized double bedsheet royal",
    "stainless steel kitchen storage container set",
    "analog wrist watch for men leather strap",
    "baby care wipes gentle for sensitive skin",
] * 5


def test_grille_ecarte_les_representations_identiques():
    """Un plafond de vocabulaire qui ne mord pas ne crée pas de configuration."""
    retenues = _grille_tfidf_utile(TEXTES)
    signatures = [(p["ngram_range"], p["min_df"], taille) for p, taille in retenues]
    assert len(signatures) == len(set(signatures)), "deux configurations identiques retenues"

    # Sur ce corpus minuscule, aucun des trois plafonds ne mord : il ne doit
    # rester qu'une configuration par couple (ngram_range, min_df).
    couples = {(p["ngram_range"], p["min_df"]) for p, _ in retenues}
    assert len(retenues) == len(couples)


def test_effets_marginaux_moyennent_les_autres_axes():
    table = pd.DataFrame(
        [
            {"axe": "a", "autre": "x", "F1": 0.90},
            {"axe": "a", "autre": "y", "F1": 0.80},
            {"axe": "b", "autre": "x", "F1": 0.70},
            {"axe": "b", "autre": "y", "F1": 0.60},
        ]
    )
    marges = _effets_marginaux(table, "F1", ["axe", "autre"])

    par_valeur = dict(zip(marges["Valeur"], marges["Moyenne"], strict=True))
    assert par_valeur["a"] == 0.85
    assert par_valeur["b"] == 0.65
    assert par_valeur["x"] == 0.80

    # L'axe qui sépare le plus doit remonter en tête : c'est ce classement que
    # le rapport lit pour dire quel hyperparamètre compte.
    assert marges.iloc[0]["Hyperparamètre"] == "axe"
    etendues = dict(zip(marges["Hyperparamètre"], marges["Étendue de l'axe"], strict=True))
    assert etendues["axe"] == 0.20
    assert etendues["autre"] == 0.10
