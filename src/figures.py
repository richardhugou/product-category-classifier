"""Les figures du projet.

    python -m src.figures

Chaque figure porte son titre : elles sont conçues pour être insérées telles
quelles dans un support, sans légende ajoutée par-dessus.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GOOD = "#0ca30c"  # statut : franchit le seuil
CRITICAL = "#d03b3b"  # statut : ne le franchit pas
GRID = "#e0dfdb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)



BLEU_CLAIR, BLEU, BLEU_FONCE = "#cde2fb", "#2a78d6", "#0d366b"


def fig3() -> None:
    """La chaîne de transformation, avec un article réel qui la traverse."""
    from matplotlib.patches import FancyBboxPatch

    etapes = [
        (
            "1 · Description brute",
            "55 mots, texte commercial libre",
            "« Maxima 07034LMLI Attivo Analog Watch – For Women – Buy Maxima…\n"
            "Rs.641 in India Only at Flipkart.com. Brass Case, Buckle Clasp… »",
        ),
        (
            "2 · Normalisation",
            "minuscules · ponctuation · mots-outils anglais retirés",
            "maxima · 07034lmli · attivo · analog · watch · women · buy …    → 85 jetons retenus",
        ),
        (
            "3 · Vocabulaire",
            "n-grammes 1–2 · terme vu au moins 2 fois · plafond 5 000",
            "4 532 termes retenus sur l'ensemble du corpus d'entraînement",
        ),
        (
            "4 · Pondération TF-IDF",
            "fréquence sublinéaire × rareté dans le corpus",
            "vecteur creux : 61 termes non nuls sur 4 532\n"
            "attivo 0,33   ·   attivo analog 0,33   ·   maxima 0,25   ·   watch women 0,21",
        ),
        (
            "5 · Classification",
            "perceptron 128 → 64 → 7, sortie softmax",
            "Watches 97,7 %   ·   Home Decor 0,8 %   ·   Beauty 0,7 %      → catégorie réelle : Watches",
        ),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(etapes) * 2 + 0.6)
    ax.axis("off")

    for k, (titre, methode, contenu) in enumerate(etapes):
        y = (len(etapes) - 1 - k) * 2 + 0.5
        ax.add_patch(
            FancyBboxPatch(
                (0.15, y),
                9.7,
                1.45,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=BLEU_CLAIR,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.15, y),
                0.075,
                1.45,
                boxstyle="square,pad=0",
                facecolor=BLEU_FONCE,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            0.42, y + 1.12, titre, fontsize=12, fontweight="bold", color=BLEU_FONCE, va="center"
        )
        ax.text(0.42, y + 0.82, methode, fontsize=9.5, color=INK_2, va="center", style="italic")
        ax.text(0.42, y + 0.36, contenu, fontsize=10, color=INK, va="center", linespacing=1.5)
        if k < len(etapes) - 1:
            ax.annotate(
                "",
                xy=(5, y - 0.48),
                xytext=(5, y - 0.07),
                arrowprops=dict(arrowstyle="-|>", color=BLEU, lw=2),
            )

    ax.set_title(
        "De la fiche produit à la catégorie — la chaîne de transformation",
        fontsize=14,
        fontweight="bold",
        pad=12,
        loc="left",
        x=0.015,
    )
    fig.text(
        0.015,
        0.015,
        "Un article du jeu de test suivi de bout en bout. Chaque étape est ajustée sur les "
        "735 articles d'entraînement seulement, jamais sur le test.",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(REPORTS / "fig3_transformations.png", dpi=200)
    print("→ reports/fig3_transformations.png")


def fig4() -> None:
    """Le jeu d'apprentissage : équilibre des classes et information disponible."""
    from src.pipeline import LABEL_COL, TEXT_COL, load, split

    df = load()
    tr, va, te = split(df)
    n = df[LABEL_COL].value_counts().sort_index()
    long = df.assign(n=df[TEXT_COL].str.split().str.len())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    a1.barh(n.index[::-1], n.values[::-1], color=BLEU, height=0.62, zorder=3)
    for i, v in enumerate(n.values[::-1]):
        a1.text(v - 8, i, str(v), va="center", ha="right", color="#ffffff", fontsize=10)
    a1.set_xlim(0, 175)
    a1.set_xlabel("articles")
    a1.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    a1.set_title(
        "7 catégories, 150 articles chacune — équilibre parfait",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    ordre = long.groupby(LABEL_COL)["n"].median().sort_values().index
    bp = a2.boxplot(
        [long.loc[long[LABEL_COL] == c, "n"] for c in ordre],
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    for b in bp["boxes"]:
        b.set(facecolor=BLEU_CLAIR, edgecolor=BLEU, linewidth=1.4)
    for part in ("whiskers", "caps"):
        for w in bp[part]:
            w.set(color=BLEU, linewidth=1.2)
    for m in bp["medians"]:
        m.set(color=BLEU_FONCE, linewidth=2)
    a2.set_yticklabels(ordre, fontsize=9.5)
    a2.set_xlabel("longueur de la description — mots")
    a2.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    a2.set_title(
        "L'information disponible, elle, ne l'est pas",
        fontsize=11.5,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    fig.text(
        0.008,
        0.015,
        f"Découpe stratifiée 70 / 15 / 15, graine 42 — identique à 2024 : "
        f"{len(tr)} entraînement · {len(va)} validation · {len(te)} test.  "
        "Médiane globale 44 mots ; 24 pour Home Furnishing, 88 pour Kitchen & Dining.",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(REPORTS / "fig4_donnees.png", dpi=200)
    print("→ reports/fig4_donnees.png")


def toutes() -> None:
    """Regénère les figures descriptives."""
    fig3()
    fig4()


if __name__ == "__main__":
    toutes()
