"""Profilage du jeu de données — les chiffres de l'audit qualité.

    python scripts/profile_data.py

Six dimensions inspirées du référentiel DAMA : complétude, unicité, validité,
cohérence, exactitude, actualité. Seules les quatre premières se testent
automatiquement ; les deux dernières relèvent du jugement et sont commentées.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src.pipeline import CSV, LABEL_COL, TEXT_COL, load, split  # noqa: E402


def main() -> None:
    brut = pd.read_csv(CSV)
    df = load()

    print(f"Fichier   : {CSV.name}")
    print(f"Dimensions: {brut.shape[0]} lignes × {brut.shape[1]} colonnes\n")

    print("— COMPLÉTUDE —")
    manquants = brut.isna().sum()
    manquants = manquants[manquants > 0].sort_values(ascending=False)
    if manquants.empty:
        print("  aucun champ manquant")
    for col, n in manquants.items():
        print(f"  {col:24s} {n:4d} manquants ({n / len(brut):.0%})")

    print("\n— UNICITÉ —")
    print(f"  doublons uniq_id      : {brut['uniq_id'].duplicated().sum()}")
    print(f"  doublons description  : {brut[TEXT_COL].duplicated().sum()}")

    print("\n— VALIDITÉ —")
    n = df[LABEL_COL].value_counts().sort_index()
    print(f"  {n.nunique()} effectif(s) distinct(s) sur {len(n)} catégories")
    for cat, v in n.items():
        print(f"  {cat:28s} {v}")

    print("\n— COHÉRENCE —")
    print(f"  lignes avec une image référencée : {brut['image'].notna().sum()} / {len(brut)}")

    print("\n— INFORMATION DISPONIBLE —")
    long = df[TEXT_COL].str.split().str.len()
    print(
        f"  longueur des descriptions (mots) : min {long.min()} · médiane {long.median():.0f} "
        f"· max {long.max()} · moyenne {long.mean():.1f}"
    )
    med = df.assign(n=long).groupby(LABEL_COL)["n"].median().sort_values()
    print(
        f"  médiane par catégorie : de {med.iloc[0]:.0f} ({med.index[0]}) "
        f"à {med.iloc[-1]:.0f} ({med.index[-1]})"
    )

    print("\n— CONSTAT : le champ `brand` est écarté —")
    absent = brut.assign(nb=brut["brand"].isna()).groupby(df[LABEL_COL])["nb"].sum()
    for cat, v in absent.sort_values(ascending=False).items():
        print(f"  {cat:28s} {int(v):3d} / 150 sans marque")
    print("  L'absence de marque n'est pas aléatoire : elle prédit la catégorie.")
    print("  Utiliser ce champ ferait fuiter la cible. Il est exclu du modèle.")

    tr, va, te = split(df)
    print(f"\n— DÉCOUPE —\n  {len(tr)} entraînement · {len(va)} validation · {len(te)} test")


if __name__ == "__main__":
    main()
