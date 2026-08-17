"""Chargement, découpe et vérité terrain : source unique pour le benchmark et l'application.

Découpe 70 / 15 / 15 stratifiée, graine 42. Aucune logique métier ailleurs :
tous les scripts et carnets appellent ce module, ce qui garantit que leurs
chiffres se comparent entre eux.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "flipkart_com-ecommerce_sample_1050.csv"

SEED = 42
TEXT_COL = "description"
LABEL_COL = "category"


def _first_level(tree: str) -> str:
    """Premier niveau de l'arbre de catégories, seule cible retenue (7 classes)."""
    return re.sub(r'^\["?', "", str(tree)).split(">>")[0].strip()


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df[LABEL_COL] = df["product_category_tree"].apply(_first_level)
    return df[["uniq_id", "product_name", TEXT_COL, LABEL_COL, "image"]]


def split(df: pd.DataFrame):
    """70 / 15 / 15 stratifié. Renvoie (train, val, test)."""
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df[LABEL_COL])
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp[LABEL_COL])
    return train, val, test


if __name__ == "__main__":
    d = load()
    tr, va, te = split(d)
    print(f"total {len(d)} · train {len(tr)} · val {len(va)} · test {len(te)}")
    print(d[LABEL_COL].value_counts().to_string())
