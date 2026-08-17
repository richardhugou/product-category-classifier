"""Optimisation des hyperparamètres, et critique de la méthode d'optimisation.

Deux modes, et la comparaison des deux est le vrai sujet de ce script.

`--holdout` sélectionne sur les 157 articles de validation. C'est la méthode
naïve, celle qu'on écrit en premier. Elle est conservée ici parce qu'elle
échoue de façon instructive : sur un jeu de cette taille, un article vaut
0,64 point de F1, et le classement est dominé par le bruit.

Par défaut, la sélection se fait par **validation croisée à 5 blocs sur les
892 articles d'entraînement et de validation réunis**. Chaque configuration
est jugée sur cinq découpes différentes, ce qui donne une moyenne et un
écart-type, donc un moyen de savoir si un écart veut dire quelque chose.

Dans les deux cas, le jeu de test reste fermé jusqu'à la toute fin et n'est
ouvert qu'une fois, avec la seule configuration retenue.

Le script rapporte aussi les **effets marginaux** : la performance moyenne de
chaque valeur d'hyperparamètre, tous les autres réglages confondus. C'est la
bonne façon de lire une exploration. La première ligne d'un classement dit ce
qui a eu de la chance ; la moyenne par valeur dit ce qui compte.

    python optimize.py --images             # validation croisée (recommandé)
    python optimize.py --images --holdout   # sélection naïve, pour comparaison
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluate import couverture_au_seuil, mesurer  # noqa: E402
from src.fusion import normaliser  # noqa: E402
from src.pipeline import LABEL_COL, SEED, TEXT_COL, load, split  # noqa: E402

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
N_BLOCS = 5

# Grille de la représentation. Plusieurs combinaisons produisent le même
# vocabulaire : un plafond de 10 000 termes ne change rien quand le corpus n'en
# produit que 4 679. Ces doublons sont écartés à la construction de la grille.
GRILLE_TFIDF = [
    {"max_features": mf, "ngram_range": ng, "min_df": md}
    for mf, ng, md in itertools.product([5000, 10000, None], [(1, 1), (1, 2)], [1, 2])
]

# Grille de la tête. `alpha` est la régularisation L2.
GRILLE_MLP = [
    {"hidden_layer_sizes": h, "alpha": a}
    for h, a in itertools.product([(128, 64), (256,), (512, 256)], [1e-4, 1e-3, 1e-2])
]

# Poids du bloc image avant concaténation. 1,0 est le réglage implicite de la
# fusion non optimisée : normaliser les deux blocs leur donne le même poids.
GRILLE_POIDS_IMAGE = [0.25, 0.5, 1.0, 2.0, 4.0]


def _tfidf(params: dict) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, **params)


def _mlp(params: dict) -> MLPClassifier:
    return MLPClassifier(max_iter=400, early_stopping=True, random_state=SEED, **params)


def _grille_tfidf_utile(textes: list[str]) -> list[tuple[dict, int]]:
    """Écarte les configurations qui produisent un vocabulaire déjà exploré."""
    vus, retenues = set(), []
    for p in GRILLE_TFIDF:
        taille = len(_tfidf(p).fit(textes).vocabulary_)
        cle = (p["ngram_range"], p["min_df"], taille)
        if cle not in vus:
            vus.add(cle)
            retenues.append((p, taille))
    return retenues


def _par_representation(table: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Performance moyenne de chaque représentation, à conception équilibrée.

    À préférer aux effets marginaux pour tout ce qui touche la représentation.
    La déduplication de la grille rend le plan déséquilibré : `max_features`
    illimité ne survit que là où il change quelque chose, c'est-à-dire sur les
    seules configurations à bigrammes : si bien que sa moyenne marginale est
    confondue avec l'effet des bigrammes. Ici chaque représentation porte le
    même nombre de configurations de tête, donc les moyennes se comparent.
    """
    groupes = table.groupby(["ngram_range", "min_df", "vocabulaire"])[colonne]
    return (
        groupes.agg(Moyenne="mean", Maximum="max", Configurations="count")
        .round(4)
        .reset_index()
        .sort_values("Moyenne", ascending=False)
    )


def _effets_marginaux(table: pd.DataFrame, colonne: str, axes: list[str]) -> pd.DataFrame:
    """Performance moyenne par valeur d'hyperparamètre, les autres confondus.

    Valable tant que le plan est équilibré sur l'axe considéré : c'est le cas
    des réglages de la tête, qui apparaissent le même nombre de fois. Pour la
    représentation, voir `_par_representation`.
    """
    lignes = []
    for axe in axes:
        for valeur, groupe in table.groupby(axe, dropna=False):
            lignes.append(
                {
                    "Hyperparamètre": axe,
                    "Valeur": str(valeur),
                    "Moyenne": round(groupe[colonne].mean(), 4),
                    "Minimum": round(groupe[colonne].min(), 4),
                    "Maximum": round(groupe[colonne].max(), 4),
                    "Configurations": len(groupe),
                }
            )
    table_e = pd.DataFrame(lignes)
    etendues = table_e.groupby("Hyperparamètre")["Moyenne"].agg(lambda s: s.max() - s.min())
    table_e["Étendue de l'axe"] = table_e["Hyperparamètre"].map(etendues).round(4)
    return table_e.sort_values(["Étendue de l'axe", "Moyenne"], ascending=[False, False])


def _score_texte_cv(textes, y, p_vec: dict, p_mlp: dict) -> tuple[float, float]:
    """F1 macro moyenne et écart-type sur 5 blocs. Le vectoriseur est réajusté
    à chaque bloc, sur la partie entraînement du bloc seulement."""
    kf = StratifiedKFold(n_splits=N_BLOCS, shuffle=True, random_state=SEED)
    scores = []
    textes = np.asarray(textes)
    for i_tr, i_va in kf.split(textes, y):
        vec = _tfidf(p_vec)
        Xtr = vec.fit_transform(textes[i_tr])
        Xva = vec.transform(textes[i_va])
        clf = _mlp(p_mlp).fit(Xtr, y[i_tr])
        scores.append(f1_score(y[i_va], clf.predict(Xva), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def _score_fusion_cv(textes, images, y, p_vec: dict, poids: float, p_mlp: dict):
    kf = StratifiedKFold(n_splits=N_BLOCS, shuffle=True, random_state=SEED)
    scores = []
    textes = np.asarray(textes)
    for i_tr, i_va in kf.split(textes, y):
        vec = _tfidf(p_vec)
        Ftr = hstack([vec.fit_transform(textes[i_tr]), csr_matrix(images[i_tr] * poids)]).tocsr()
        Fva = hstack([vec.transform(textes[i_va]), csr_matrix(images[i_va] * poids)]).tocsr()
        clf = _mlp(p_mlp).fit(Ftr, y[i_tr])
        scores.append(f1_score(y[i_va], clf.predict(Fva), average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main(avec_images: bool, holdout: bool) -> None:
    REPORTS.mkdir(exist_ok=True)
    suffixe = "holdout" if holdout else "cv"
    df = load()
    train, val, test = split(df)

    enc = LabelEncoder().fit(df[LABEL_COL])
    etiquettes = list(enc.classes_)
    y_tr, y_va, y_te = (enc.transform(d[LABEL_COL]) for d in (train, val, test))
    txt_tr, txt_va, txt_te = (d[TEXT_COL].tolist() for d in (train, val, test))

    # En validation croisée, entraînement et validation sont réunis : la
    # découpe interne s'en charge, et la sélection porte sur 892 articles.
    if holdout:
        txt_sel, y_sel = txt_tr, y_tr
        print(f"Sélection sur les {len(y_va)} articles de validation (méthode naïve).")
    else:
        txt_sel = txt_tr + txt_va
        y_sel = np.concatenate([y_tr, y_va])
        print(f"Sélection par validation croisée à {N_BLOCS} blocs sur {len(y_sel)} articles.")
    print(f"Le test ({len(y_te)} articles) reste fermé jusqu'à la fin.\n")

    grille_vec = _grille_tfidf_utile(txt_sel)
    print(f"{len(GRILLE_TFIDF)} configurations de représentation, {len(grille_vec)} distinctes")

    # ------------------------------------------------------------------ texte
    t0 = time.perf_counter()
    lignes = []
    for p_vec, taille in grille_vec:
        if holdout:
            vec = _tfidf(p_vec)
            Xtr, Xva = vec.fit_transform(txt_tr), vec.transform(txt_va)
        for p_mlp in GRILLE_MLP:
            if holdout:
                clf = _mlp(p_mlp).fit(Xtr, y_tr)
                moyenne = float(f1_score(y_va, clf.predict(Xva), average="macro"))
                ecart = 0.0
            else:
                moyenne, ecart = _score_texte_cv(txt_sel, y_sel, p_vec, p_mlp)
            lignes.append(
                {
                    "max_features": p_vec["max_features"] or "tous",
                    "ngram_range": str(p_vec["ngram_range"]),
                    "min_df": p_vec["min_df"],
                    "vocabulaire": taille,
                    "hidden_layer_sizes": str(p_mlp["hidden_layer_sizes"]),
                    "alpha": p_mlp["alpha"],
                    "F1 macro": round(moyenne, 4),
                    "Écart-type": round(ecart, 4),
                }
            )
        print(f"  {p_vec['ngram_range']} min_df={p_vec['min_df']} · vocabulaire {taille}")

    table = pd.DataFrame(lignes).sort_values("F1 macro", ascending=False)
    table.to_csv(REPORTS / f"optimisation_texte_{suffixe}.csv", index=False)

    axes_texte = ["ngram_range", "min_df", "max_features", "hidden_layer_sizes", "alpha"]
    marges = _effets_marginaux(table, "F1 macro", axes_texte)
    marges.to_csv(REPORTS / f"effets_marginaux_texte_{suffixe}.csv", index=False)
    representations = _par_representation(table, "F1 macro")
    representations.to_csv(REPORTS / f"representations_{suffixe}.csv", index=False)

    m = table.iloc[0]
    print(f"\n  {len(table)} combinaisons en {time.perf_counter() - t0:.0f} s")
    print(f"  meilleure : F1 {m['F1 macro']:.4f} ± {m['Écart-type']:.4f}")
    print(f"  étendue du classement : {table['F1 macro'].max() - table['F1 macro'].min():.4f}")
    print("\n  Par représentation : plan équilibré, moyennes comparables :")
    print(representations.to_string(index=False))
    print("\n  Effets marginaux (à ne lire que pour les axes de la tête) :")
    print(marges.to_string(index=False))

    p_vec_best = {
        "max_features": None if m["max_features"] == "tous" else int(m["max_features"]),
        "ngram_range": (1, 2) if m["ngram_range"] == "(1, 2)" else (1, 1),
        "min_df": int(m["min_df"]),
    }
    p_mlp_best = {
        "hidden_layer_sizes": {"(128, 64)": (128, 64), "(256,)": (256,), "(512, 256)": (512, 256)}[
            m["hidden_layer_sizes"]
        ],
        "alpha": float(m["alpha"]),
    }

    # Réapprentissage final sur tout ce qui a servi à sélectionner, puis test.
    vec = _tfidf(p_vec_best)
    Xsel, Xte = vec.fit_transform(txt_sel), vec.transform(txt_te)
    clf = _mlp(p_mlp_best).fit(Xsel, y_sel)
    resultats = [mesurer("Texte optimisé", y_te, clf.predict(Xte), etiquettes, 0.0, 0.0, 0.0)]
    print(
        f"\n  TEST : F1 macro {resultats[0]['F1 macro']:.4f} · min {resultats[0]['F1 classe min']}"
    )

    seuils = {}

    # ----------------------------------------------------------------- fusion
    if avec_images:
        from src.images import features

        print("\nFusion : poids du bloc image")
        X_img, _ = features(df["uniq_id"])
        par_id = dict(zip(df["uniq_id"], X_img, strict=True))
        img = {
            nom: normaliser(np.vstack([par_id[u] for u in d["uniq_id"]]))
            for nom, d in (("tr", train), ("va", val), ("te", test))
        }
        img_sel = img["tr"] if holdout else np.vstack([img["tr"], img["va"]])

        lignes = []
        for poids in GRILLE_POIDS_IMAGE:
            for p_mlp in GRILLE_MLP:
                if holdout:
                    Ftr = hstack([Xsel, csr_matrix(img["tr"] * poids)]).tocsr()
                    Fva = hstack([vec.transform(txt_va), csr_matrix(img["va"] * poids)]).tocsr()
                    c = _mlp(p_mlp).fit(Ftr, y_tr)
                    moyenne, ecart = float(f1_score(y_va, c.predict(Fva), average="macro")), 0.0
                else:
                    moyenne, ecart = _score_fusion_cv(
                        txt_sel, img_sel, y_sel, p_vec_best, poids, p_mlp
                    )
                lignes.append(
                    {
                        "poids image": poids,
                        "hidden_layer_sizes": str(p_mlp["hidden_layer_sizes"]),
                        "alpha": p_mlp["alpha"],
                        "F1 macro": round(moyenne, 4),
                        "Écart-type": round(ecart, 4),
                    }
                )
            print(f"  poids {poids}")

        table_f = pd.DataFrame(lignes).sort_values("F1 macro", ascending=False)
        table_f.to_csv(REPORTS / f"optimisation_fusion_{suffixe}.csv", index=False)
        marges_f = _effets_marginaux(
            table_f, "F1 macro", ["poids image", "hidden_layer_sizes", "alpha"]
        )
        marges_f.to_csv(REPORTS / f"effets_marginaux_fusion_{suffixe}.csv", index=False)

        mf = table_f.iloc[0]
        print(f"\n  meilleure : F1 {mf['F1 macro']:.4f} ± {mf['Écart-type']:.4f}")
        print("\n  Effets marginaux :")
        print(marges_f.to_string(index=False))

        poids = float(mf["poids image"])
        p_mlp_f = {
            "hidden_layer_sizes": {
                "(128, 64)": (128, 64),
                "(256,)": (256,),
                "(512, 256)": (512, 256),
            }[mf["hidden_layer_sizes"]],
            "alpha": float(mf["alpha"]),
        }
        Fsel = hstack([Xsel, csr_matrix(img_sel * poids)]).tocsr()
        Fte = hstack([Xte, csr_matrix(img["te"] * poids)]).tocsr()
        clf_f = _mlp(p_mlp_f).fit(Fsel, y_sel)
        resultats.append(
            mesurer("Fusion optimisée", y_te, clf_f.predict(Fte), etiquettes, 0.0, 0.0, 0.0)
        )
        print(
            f"  TEST : F1 macro {resultats[-1]['F1 macro']:.4f} · min {resultats[-1]['F1 classe min']}"
        )

        vrai_te = [etiquettes[i] for i in y_te]
        seuils["Fusion optimisée"] = couverture_au_seuil(
            clf_f.predict_proba(Fte), vrai_te, etiquettes
        ).to_dict("records")

    # ---------------------------------------------------------------- sorties
    for r in resultats:
        r.pop("_par_classe", None)
        for cle in ("Entraînement (s)", "Inférence (ms/article)", "Empreinte (Mo)"):
            r.pop(cle, None)
    (REPORTS / f"optimisation_{suffixe}.json").write_text(
        json.dumps(
            {"methode": suffixe, "test": resultats, "seuils": seuils}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"\n→ reports/optimisation_{suffixe}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", action="store_true", help="optimise aussi la fusion")
    p.add_argument("--holdout", action="store_true", help="sélection naïve sur la validation")
    a = p.parse_args()
    main(a.images, a.holdout)
