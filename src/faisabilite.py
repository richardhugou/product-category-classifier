"""Projeter, segmenter, mesurer : l'étude de faisabilité proprement dite.

La question n'est pas de prédire mais de savoir si les catégories existent
déjà dans les représentations. On y répond en deux temps, comme le demande la
mission : d'abord une projection en deux dimensions qu'on regarde, ensuite une
mesure qui confirme ou dément ce qu'on a cru voir.

La mesure est l'indice de Rand ajusté. Il compare deux partitions d'un même
ensemble, ici les vraies catégories et les groupes formés sans elles, et
vaut 0 quand l'accord n'excède pas le hasard, 1 quand les deux coïncident.
L'ajustement est ce qui compte : sur sept groupes de tailles voisines, un
appariement aléatoire produit déjà un accord apparent non négligeable.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from src.pipeline import SEED

DIM_ACP = 50
PERPLEXITE = 30


def _preparer(X: np.ndarray) -> np.ndarray:
    """Ramène chaque produit à une longueur de 1, et rien d'autre.

    Deux tentations sont écartées ici. Ne rien faire laisserait les
    descriptions longues peser plus lourd que les courtes, alors que la
    longueur d'un texte ne dit pas sa catégorie. Standardiser chaque dimension,
    à l'inverse, donnerait à un terme apparu trois fois dans tout le corpus le
    même poids qu'à un terme structurant : sur cinq mille dimensions dont la
    plupart sont vides, cela revient à amplifier le bruit.

    La normalisation par ligne évite les deux. Elle rend aussi la distance
    euclidienne équivalente à la distance cosinus, qui est la bonne mesure de
    proximité aussi bien pour un sac de mots que pour un plongement.
    """
    return normalize(X)


def projeter(X: np.ndarray) -> np.ndarray:
    """Réduction en deux dimensions : ACP puis t-SNE.

    L'ACP fait le gros du chemin à moindre coût et débruite ; t-SNE se charge
    ensuite de la mise en plan, en cherchant à préserver les voisinages plutôt
    que les distances globales. C'est pour cela qu'on lit la proximité des
    points sur ces graphiques, jamais l'échelle ni la distance entre îlots.
    """
    Xn = _preparer(X)
    composantes = min(DIM_ACP, Xn.shape[1], Xn.shape[0] - 1)
    Xr = PCA(n_components=composantes, random_state=SEED).fit_transform(Xn)
    return TSNE(
        n_components=2, perplexity=PERPLEXITE, init="pca", random_state=SEED, max_iter=1000
    ).fit_transform(Xr)


def segmenter(X: np.ndarray, k: int = 7) -> np.ndarray:
    """K-means en k groupes, sans jamais voir les étiquettes."""
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)


def accord(categories, groupes) -> float:
    """Indice de Rand ajusté entre les vraies catégories et les groupes."""
    return float(adjusted_rand_score(categories, groupes))


def etudier(X: np.ndarray, categories) -> dict:
    """Projette, segmente des deux façons, et mesure.

    L'accord est calculé deux fois. Sur la projection, parce que c'est elle
    qu'on a regardée et que la mission demande de confirmer l'analyse visuelle.
    Sur la représentation complète, parce que t-SNE déforme et qu'il serait
    malhonnête de ne rapporter que le chiffre le plus flatteur des deux.
    """
    plan = projeter(X)
    return {
        "projection": plan,
        "groupes_projection": (gp := segmenter(plan)),
        "groupes_complets": (gc := segmenter(_preparer(X))),
        "ARI projection": round(accord(categories, gp), 4),
        "ARI représentation complète": round(accord(categories, gc), 4),
        "dimensions": int(X.shape[1]),
    }
