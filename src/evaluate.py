"""Métriques et arbitrage.

Le seuil métier est une constante du module : il est fixé **avant** toute
mesure, et le code entier s'y réfère. Une tolérance dérivée du résultat obtenu
place le modèle à la limite d'acceptation par construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SEUIL_F1_METIER = 0.90


def mesurer(
    nom: str,
    y_vrai,
    y_pred,
    etiquettes: list[str],
    entrainement_s: float,
    inference_ms: float,
    empreinte_mo: float,
) -> dict:
    """Une ligne du tableau de comparaison."""
    par_classe = f1_score(y_vrai, y_pred, average=None, labels=range(len(etiquettes)))
    f1m = float(f1_score(y_vrai, y_pred, average="macro"))
    return {
        "Modèle": nom,
        "F1 macro": round(f1m, 4),
        "Accuracy": round(float(accuracy_score(y_vrai, y_pred)), 4),
        "F1 classe min": round(float(par_classe.min()), 4),
        "Entraînement (s)": round(entrainement_s, 2),
        "Inférence (ms/article)": round(inference_ms, 2),
        "Empreinte (Mo)": round(empreinte_mo, 1),
        "Seuil 0,90": "oui" if f1m >= SEUIL_F1_METIER else "non",
        "_par_classe": dict(zip(etiquettes, [round(float(v), 3) for v in par_classe], strict=True)),
    }


def couverture_au_seuil(
    proba: np.ndarray, y_vrai, etiquettes: list[str], seuils=(0.5, 0.6, 0.7, 0.8, 0.9)
) -> pd.DataFrame:
    """Compromis couverture / erreurs : l'abstention sous seuil, chiffrée.

    Sans son taux d'erreur associé, une couverture ne veut rien dire : les deux
    colonnes se lisent ensemble.
    """
    pred = np.asarray([etiquettes[i] for i in proba.argmax(1)])
    conf = proba.max(1)
    vrai = np.asarray(list(y_vrai))

    lignes = []
    for s in seuils:
        retenus = conf >= s
        n = int(retenus.sum())
        erreurs = int((pred[retenus] != vrai[retenus]).sum()) if n else 0
        lignes.append(
            {
                "Seuil": s,
                "Couverture": round(float(retenus.mean()), 4),
                "Articles retenus": n,
                "Erreurs": erreurs,
                "Taux d'erreur": round(erreurs / n, 4) if n else None,
            }
        )
    return pd.DataFrame(lignes)
