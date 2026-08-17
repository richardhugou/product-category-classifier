"""La découpe est le socle de toute mesure : elle est testée avant le reste."""

from __future__ import annotations

import pytest

from src.pipeline import LABEL_COL, TEXT_COL, load, split


@pytest.fixture(scope="module")
def df():
    return load()


def test_volume_et_classes(df):
    assert len(df) == 1050
    assert df[LABEL_COL].nunique() == 7


def test_classes_equilibrees(df):
    """150 articles par catégorie : l'équilibre est parfait, et on le vérifie."""
    assert set(df[LABEL_COL].value_counts()) == {150}


def test_aucune_description_manquante(df):
    assert df[TEXT_COL].notna().all()


def test_identifiants_uniques(df):
    assert df["uniq_id"].is_unique


def test_tailles_de_decoupe(df):
    train, val, test = split(df)
    assert (len(train), len(val), len(test)) == (735, 157, 158)
    assert len(train) + len(val) + len(test) == len(df)


def test_decoupes_disjointes(df):
    """Aucun article ne peut se trouver dans deux plis : c'est LE test anti-fuite."""
    train, val, test = split(df)
    a, b, c = (set(p["uniq_id"]) for p in (train, val, test))
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)
    assert len(a | b | c) == len(df)


def test_decoupe_reproductible(df):
    """Graine fixe : deux appels donnent exactement la même partition."""
    premier = [set(p["uniq_id"]) for p in split(df)]
    second = [set(p["uniq_id"]) for p in split(df)]
    assert premier == second


def test_decoupe_stratifiee(df):
    """Chaque pli contient les 7 catégories, en proportions comparables."""
    for pli in split(df):
        parts = pli[LABEL_COL].value_counts(normalize=True)
        assert len(parts) == 7
        assert parts.max() - parts.min() < 0.02
