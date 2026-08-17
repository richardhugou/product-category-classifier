"""Seuil de confiance : quelle part du catalogue peut être classée sans revue humaine.

Le modèle retenu rend sept probabilités par article. Au-dessus d'un seuil, la
proposition est appliquée telle quelle ; en dessous, l'article part en revue.
Deux indicateurs en découlent, l'un métier, l'autre technique :

    taux d'automatisation   part des articles traités sans intervention
    précision sur la part automatisée   part correcte parmi ceux-là

Protocole identique au reste de l'étude : le seuil est choisi sur la validation,
puis le couple d'indicateurs est mesuré une seule fois sur le jeu de test. Le
test n'entre pas dans le choix du seuil.

    python scripts/seuil_confiance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion import concatener, normaliser  # noqa: E402
from src.images import features  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402
from src.supervise_image import tete  # noqa: E402
from src.text import vectoriseur  # noqa: E402

SORTIE = ROOT / "reports" / "seuil_confiance.csv"
GRILLE = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
# Exigence posée d'avance : sur la part automatisée, au moins 99 % de propositions
# correctes. Le seuil retenu est le plus bas qui la satisfait sur la validation.
EXIGENCE = 0.99


def _mesures(probas: np.ndarray, y: np.ndarray, seuil: float) -> dict:
    confiance = probas.max(axis=1)
    pred = probas.argmax(axis=1)
    auto = confiance >= seuil
    n_auto = int(auto.sum())
    return {
        "seuil": seuil,
        "automatisés": n_auto,
        "total": len(y),
        "taux d'automatisation": n_auto / len(y),
        "précision sur la part automatisée": (
            float((pred[auto] == y[auto]).mean()) if n_auto else float("nan")
        ),
        "erreurs automatisées": int((pred[auto] != y[auto]).sum()),
        "en revue humaine": len(y) - n_auto,
    }


def main() -> int:
    df = load()
    train, validation, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])

    vec = vectoriseur().fit(train[TEXT_COL])
    X_img, _ = features(df["uniq_id"])
    par_id = dict(zip(df["uniq_id"], X_img, strict=True))

    def X(part: pd.DataFrame) -> np.ndarray:
        dino = np.vstack([par_id[u] for u in part["uniq_id"]])
        return concatener(vec.transform(part[TEXT_COL]), normaliser(dino))

    clf = tete().fit(X(train), enc.transform(train[LABEL_COL]))

    jeux = {}
    for nom, part in (("validation", validation), ("test", test)):
        jeux[nom] = (clf.predict_proba(X(part)), enc.transform(part[LABEL_COL]))

    lignes = []
    for nom, (probas, y) in jeux.items():
        for seuil in GRILLE:
            lignes.append({"jeu": nom, **_mesures(probas, y, seuil)})
    table = pd.DataFrame(lignes)

    val = table[table["jeu"] == "validation"]
    eligibles = val[val["précision sur la part automatisée"] >= EXIGENCE]
    if eligibles.empty:
        print(f"Aucun seuil de la grille n'atteint {EXIGENCE:.0%} sur la validation.")
        retenu = float(val.loc[val["précision sur la part automatisée"].idxmax(), "seuil"])
    else:
        retenu = float(eligibles["seuil"].min())

    table["retenu"] = table["seuil"] == retenu
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SORTIE, index=False)

    COL_TAUX = "taux d'automatisation"
    COL_PREC = "précision sur la part automatisée"
    fmt = "  {:<11s} {:>6s} {:>14s} {:>11s} {:>7s}"
    print(fmt.format("jeu", "seuil", "automatisation", "précision", "revue"))
    for _, r in table.iterrows():
        taux, prec = r[COL_TAUX], r[COL_PREC]
        print(
            fmt.format(
                str(r["jeu"]),
                format(r["seuil"], ".2f"),
                format(taux, ".3f"),
                format(prec, ".4f"),
                str(int(r["en revue humaine"])),
            )
            + (" ←" if r["retenu"] else "")
        )

    print(f"\nSeuil retenu sur la validation : {retenu:.2f} (exigence {EXIGENCE:.0%})")
    for nom in ("validation", "test"):
        r = table[(table["jeu"] == nom) & table["retenu"]].iloc[0]
        taux, prec = r[COL_TAUX], r[COL_PREC]
        erreurs, revue = int(r["erreurs automatisées"]), int(r["en revue humaine"])
        print(
            f"  {nom:<11s} automatisation {taux:.1%} · précision {prec:.4f}"
            f" · {erreurs} erreur(s) automatisée(s) · {revue} en revue"
        )
    print(f"\n→ {SORTIE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
