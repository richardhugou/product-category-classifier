"""Deux schémas du rapport : le flux existant, puis l'architecture cible.

Le rapport de conduite de projet demande une visualisation du processus en place
et un schéma de la solution proposée. Les deux sont composés ici pour rester
reproductibles et cohérents avec la charte des autres figures.

    python scripts/schemas.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

ENCRE = "#1F2A33"
BLEU = "#14486B"
GRIS = "#98A3AD"
TERRE = "#A0430F"
FILET = "#C3BFB6"
DOUX = "#EDF1F5"


def _boite(ax, x, y, larg, h, titre, ligne="", teinte=BLEU, remplissage="white", pointille=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            larg,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=teinte,
            facecolor=remplissage,
            linestyle=(0, (3, 2)) if pointille else "solid",
            zorder=3,
        )
    )
    ax.text(
        x + larg / 2,
        y + h / 2 + (0.035 if ligne else 0),
        titre,
        ha="center",
        va="center",
        fontsize=8.5,
        color=ENCRE,
        fontweight="medium",
        zorder=4,
    )
    if ligne:
        ax.text(
            x + larg / 2,
            y + h / 2 - 0.05,
            ligne,
            ha="center",
            va="center",
            fontsize=7,
            color="#4E5862",
            zorder=4,
        )


def _fleche(ax, xy1, xy2, etiquette="", teinte=FILET, dessous=False):
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=teinte,
            zorder=2,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if etiquette:
        ax.text(
            (xy1[0] + xy2[0]) / 2,
            (xy1[1] + xy2[1]) / 2 + (-0.035 if dessous else 0.035),
            etiquette,
            ha="center",
            va="top" if dessous else "bottom",
            fontsize=6.8,
            color="#7C858E",
            zorder=4,
        )


def _cadre(largeur, hauteur):
    fig, ax = plt.subplots(figsize=(largeur, hauteur), dpi=200)
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.03, 1.0)
    ax.axis("off")
    return fig, ax


def flux_actuel() -> Path:
    """Le processus en place : la catégorie est déclarée, jamais vérifiée."""
    fig, ax = _cadre(10, 2.5)

    _boite(ax, 0.02, 0.52, 0.17, 0.30, "Vendeur", "photographie + description", GRIS)
    _boite(ax, 0.25, 0.52, 0.17, 0.30, "Formulaire", "catégorie saisie à la main", TERRE)
    _boite(ax, 0.48, 0.52, 0.17, 0.30, "Catalogue", "catégorie déclarée, non vérifiée", GRIS)
    _boite(ax, 0.71, 0.52, 0.17, 0.30, "Recherche", "filtre par catégorie", GRIS)

    for a, b in ((0.19, 0.25), (0.42, 0.48), (0.65, 0.71)):
        _fleche(ax, (a, 0.67), (b, 0.67))

    _boite(
        ax,
        0.25,
        0.10,
        0.40,
        0.24,
        "Aucun contrôle, aucune mesure de qualité",
        "ni règle commune entre vendeurs, ni indicateur de suivi",
        TERRE,
        "#FBF2ED",
        True,
    )
    _fleche(ax, (0.335, 0.52), (0.335, 0.34), teinte=TERRE)
    _fleche(ax, (0.565, 0.52), (0.565, 0.34), teinte=TERRE)

    ax.text(
        0.90,
        0.67,
        "Article mal rangé\ninvisible au filtrage",
        ha="left",
        va="center",
        fontsize=7,
        color=TERRE,
    )

    sortie = REPORTS / "fig11_flux_actuel.png"
    fig.savefig(sortie, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return sortie


def architecture_cible() -> Path:
    """La solution proposée : proposition automatique, seuil, revue humaine."""
    fig, ax = _cadre(10, 4.2)

    ax.text(0.02, 0.95, "COLLECTE", fontsize=6.8, color="#7C858E", family="monospace")
    _boite(ax, 0.02, 0.66, 0.14, 0.22, "Mise en ligne", "texte + image", GRIS)

    ax.text(
        0.21, 0.95, "SERVICE DE CATÉGORISATION", fontsize=6.8, color="#7C858E", family="monospace"
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.205, 0.38),
            0.45,
            0.54,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1,
            edgecolor=FILET,
            facecolor=DOUX,
            zorder=1,
        )
    )
    _boite(ax, 0.23, 0.66, 0.14, 0.22, "TF-IDF", "4 532 dimensions", BLEU)
    _boite(ax, 0.23, 0.42, 0.14, 0.22, "DINOv2 figé", "1 536 dimensions", BLEU)
    _boite(ax, 0.42, 0.54, 0.09, 0.22, "Fusion", "L2 · 6 068", BLEU)
    _boite(ax, 0.55, 0.54, 0.09, 0.22, "MLP 256", "7 probabilités", BLEU)

    _fleche(ax, (0.16, 0.77), (0.23, 0.77))
    _fleche(ax, (0.16, 0.72), (0.23, 0.53))
    _fleche(ax, (0.37, 0.77), (0.42, 0.69))
    _fleche(ax, (0.37, 0.53), (0.42, 0.61))
    _fleche(ax, (0.51, 0.65), (0.55, 0.65))

    ax.text(0.68, 0.95, "DÉCISION", fontsize=6.8, color="#7C858E", family="monospace")
    _boite(ax, 0.70, 0.54, 0.12, 0.22, "Seuil 0,60", "confiance maximale", TERRE)
    _fleche(ax, (0.64, 0.65), (0.70, 0.65))

    _boite(ax, 0.86, 0.70, 0.13, 0.19, "Appliqué", "85,4 % du volume", BLEU)
    _boite(ax, 0.86, 0.41, 0.13, 0.19, "Revue humaine", "14,6 % du volume", GRIS)
    _fleche(ax, (0.82, 0.69), (0.86, 0.78), "≥ 0,60", BLEU)
    _fleche(ax, (0.82, 0.61), (0.86, 0.52), "< 0,60", GRIS, dessous=True)

    ax.text(
        0.02,
        0.29,
        "SOCLE TECHNIQUE ET SURVEILLANCE",
        fontsize=6.8,
        color="#7C858E",
        family="monospace",
    )
    _boite(ax, 0.02, 0.05, 0.22, 0.19, "Conteneur + API", "artefact 37,5 Mo, authentifiée", BLEU)
    _boite(
        ax, 0.27, 0.05, 0.22, 0.19, "Intégration continue", "23 tests, rejeu depuis un clone", BLEU
    )
    _boite(
        ax,
        0.52,
        0.05,
        0.22,
        0.19,
        "Journalisation",
        "confiance, taux d'automatisation",
        BLEU,
        pointille=True,
    )
    _boite(
        ax,
        0.77,
        0.05,
        0.22,
        0.19,
        "Dérive et réentraînement",
        "sur corpus ré-étiqueté",
        BLEU,
        pointille=True,
    )

    ax.text(
        0.02, 0.00, "Trait plein : en place.  Pointillé : proposé.", fontsize=6.8, color="#7C858E"
    )

    sortie = REPORTS / "fig12_architecture_cible.png"
    fig.savefig(sortie, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return sortie


if __name__ == "__main__":
    REPORTS.mkdir(parents=True, exist_ok=True)
    for f in (flux_actuel(), architecture_cible()):
        print(f"  {f.relative_to(ROOT)}")
