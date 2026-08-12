"""Les métriques, et surtout la logique d'abstention."""

from __future__ import annotations

import numpy as np

from src.evaluate import SEUIL_F1_METIER, couverture_au_seuil, mesurer

ETIQUETTES = ["a", "b", "c"]


def test_seuil_metier_est_une_constante():
    """Il est fixé avant les mesures, donc il vit dans le code, pas dans un calcul."""
    assert SEUIL_F1_METIER == 0.90


def test_mesurer_cas_parfait():
    ligne = mesurer("parfait", [0, 1, 2], [0, 1, 2], ETIQUETTES, 1.0, 2.0, 3.0)
    assert ligne["F1 macro"] == 1.0
    assert ligne["F1 classe min"] == 1.0
    assert ligne["Seuil 0,90"] == "oui"


def test_mesurer_signale_l_echec_au_seuil():
    ligne = mesurer("faible", [0, 0, 1, 1], [0, 1, 0, 1], ETIQUETTES, 1.0, 2.0, 3.0)
    assert ligne["Seuil 0,90"] == "non"


def test_f1_classe_min_reperе_la_classe_faible():
    """Une classe systématiquement ratée doit tirer la F1 minimale à zéro."""
    ligne = mesurer("borgne", [0, 1, 2, 2], [0, 1, 0, 1], ETIQUETTES, 1.0, 2.0, 3.0)
    assert ligne["F1 classe min"] == 0.0
    assert ligne["F1 macro"] < 1.0


def test_couverture_decroit_avec_le_seuil():
    proba = np.array([[0.95, 0.03, 0.02], [0.55, 0.30, 0.15], [0.40, 0.35, 0.25]])
    t = couverture_au_seuil(proba, ["a", "a", "a"], ETIQUETTES)
    couvertures = t["Couverture"].tolist()
    assert couvertures == sorted(couvertures, reverse=True)


def test_abstention_ecarte_les_predictions_incertaines():
    """Au seuil de 0,9 seule la prédiction sûre est retenue, et elle est juste."""
    proba = np.array([[0.95, 0.03, 0.02], [0.40, 0.35, 0.25]])
    t = couverture_au_seuil(proba, ["a", "b"], ETIQUETTES, seuils=(0.9,))
    ligne = t.iloc[0]
    assert ligne["Articles retenus"] == 1
    assert ligne["Erreurs"] == 0
    assert ligne["Couverture"] == 0.5
