"""Étude de faisabilité — les sept représentations, projetées puis segmentées.

Répond à la première demande : les produits d'une même catégorie se
rapprochent-ils spontanément, une fois traduits en nombres ? On projette, on
regarde, puis on mesure l'accord entre les groupes formés sans étiquettes et
les vraies catégories.

    python faisabilite.py            # tout, depuis le cache si possible
    python faisabilite.py --texte    # les cinq représentations de texte seules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from src.faisabilite import etudier  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load  # noqa: E402
from src.representations import IMAGE, TEXTE, obtenir  # noqa: E402

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"

# Palette qualitative lisible en niveaux de gris et distinguable pour les
# principales déficiences de vision des couleurs.
PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#9d755d", "#79706e"]


def _nuage(axe, plan, valeurs, ordre, titre: str) -> None:
    for i, valeur in enumerate(ordre):
        masque = [v == valeur for v in valeurs]
        axe.scatter(
            plan[masque, 0],
            plan[masque, 1],
            s=7,
            c=PALETTE[i % len(PALETTE)],
            label=str(valeur),
            alpha=0.75,
            linewidths=0,
        )
    axe.set_title(titre, fontsize=10)
    axe.set_xticks([])
    axe.set_yticks([])
    for bord in axe.spines.values():
        bord.set_color("#d0d0d0")


def main(texte_seul: bool) -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load()
    categories = df[LABEL_COL].tolist()
    ordre = sorted(set(categories))
    entrees = {"texte": df[TEXT_COL].tolist(), "image": df["uniq_id"].tolist()}

    familles = [("texte", TEXTE)] if texte_seul else [("texte", TEXTE), ("image", IMAGE)]
    resultats, plans = [], {}

    for famille, registre in familles:
        print(f"\n{famille.upper()}")
        for nom in registre:
            X, secondes = obtenir(nom, entrees[famille])
            etude = etudier(X, categories)
            plans[nom] = etude
            resultats.append(
                {
                    "Représentation": nom,
                    "Source": famille,
                    "Dimensions": etude["dimensions"],
                    "ARI (projection 2D)": etude["ARI projection"],
                    "ARI (représentation complète)": etude["ARI représentation complète"],
                    "Calcul (s)": round(secondes, 1),
                }
            )
            print(
                f"  {nom:20s} {etude['dimensions']:6d} dim · "
                f"ARI plan {etude['ARI projection']:.3f} · "
                f"ARI complet {etude['ARI représentation complète']:.3f}"
            )

    tableau = pd.DataFrame(resultats).sort_values("ARI (projection 2D)", ascending=False)
    tableau.to_csv(REPORTS / "faisabilite.csv", index=False)

    # --- toutes les projections, colorées par la catégorie réelle
    noms = list(plans)
    colonnes = 4
    lignes = -(-len(noms) // colonnes)
    fig, axes = plt.subplots(lignes, colonnes, figsize=(4 * colonnes, 4 * lignes))
    for axe, nom in zip(axes.ravel(), noms, strict=False):
        ari = plans[nom]["ARI projection"]
        _nuage(axe, plans[nom]["projection"], categories, ordre, f"{nom} — ARI {ari:.3f}")
    for axe in axes.ravel()[len(noms) :]:
        axe.axis("off")
    poignees, etiquettes = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        poignees, etiquettes, loc="lower center", ncol=4, frameon=False, markerscale=2.5, fontsize=9
    )
    fig.suptitle("Projection des 1 050 produits — couleur : catégorie réelle", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(REPORTS / "fig5_projections.png", dpi=150)
    plt.close(fig)

    # --- la meilleure représentation : vraies catégories contre groupes trouvés
    meilleure = tableau.iloc[0]["Représentation"]
    etude = plans[meilleure]
    fig, (g, d) = plt.subplots(1, 2, figsize=(11, 5))
    _nuage(g, etude["projection"], categories, ordre, "Catégories réelles")
    _nuage(
        d,
        etude["projection"],
        etude["groupes_projection"],
        sorted(set(etude["groupes_projection"])),
        f"Groupes formés sans étiquettes — ARI {etude['ARI projection']:.3f}",
    )
    fig.suptitle(f"{meilleure} — ce que l'algorithme retrouve seul", fontsize=13)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig6_clusters.png", dpi=150)
    plt.close(fig)

    # --- l'accord, représentation par représentation
    fig, axe = plt.subplots(figsize=(8, 0.5 * len(tableau) + 1.6))
    y = range(len(tableau))
    axe.barh(list(y), tableau["ARI (projection 2D)"], color="#4c78a8", height=0.6)
    axe.set_yticks(list(y))
    axe.set_yticklabels(tableau["Représentation"])
    axe.invert_yaxis()
    axe.set_xlabel("Indice de Rand ajusté — 0 = hasard, 1 = accord parfait")
    axe.set_title("Accord entre les groupes trouvés et les catégories réelles", fontsize=12)
    for i, v in enumerate(tableau["ARI (projection 2D)"]):
        axe.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
    axe.set_xlim(0, max(0.5, tableau["ARI (projection 2D)"].max() * 1.18))
    for bord in ("top", "right"):
        axe.spines[bord].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig7_ari.png", dpi=150)
    plt.close(fig)

    print("\n" + tableau.to_string(index=False))
    print(f"\n→ {REPORTS / 'faisabilite.csv'} et fig5 à fig7")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--texte", action="store_true", help="les représentations de texte seules")
    main(p.parse_args().texte)
