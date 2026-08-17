"""Classification supervisée des images : sélection sur validation, test ouvert une fois.

L'ordre des opérations est ici la seule chose qui compte vraiment.

Les stratégies d'augmentation sont comparées sur les 157 produits de
validation. La stratégie retenue est celle qui obtient la meilleure F1 macro
sur ce jeu-là. Ce n'est qu'ensuite, une fois ce choix figé, que les 158
produits de test sont ouverts : une seule fois, avec ce seul modèle.

Cette discipline n'est pas décorative. Comparer quatre stratégies sur le test
puis retenir la meilleure ferait du score de test une mesure de la qualité de
notre sélection plutôt que de la performance du modèle. Le jeu de test cesse
d'être indépendant dès que ses résultats orientent la décision suivante, même
si aucune étiquette de test n'atteint jamais les poids du modèle.

    python supervise_image.py              # sélection sur validation, puis test
    python supervise_image.py --copies 6   # davantage de variantes augmentées
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import LABEL_COL, load, split  # noqa: E402
from src.supervise_image import extraire, tete  # noqa: E402

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
COULEURS = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]


def _f1_par_classe(y_vrai, y_pred, etiquettes: list[str]) -> dict:
    scores = f1_score(y_vrai, y_pred, average=None, labels=range(len(etiquettes)))
    return dict(zip(etiquettes, [round(float(v), 3) for v in scores], strict=True))


def main(copies: int) -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load()
    train, val, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])
    etiquettes = list(enc.classes_)
    y_tr, y_va, y_te = (enc.transform(d[LABEL_COL]) for d in (train, val, test))
    ids_tr, ids_va, ids_te = (d["uniq_id"].tolist() for d in (train, val, test))

    print(f"train {len(ids_tr)} · validation {len(ids_va)} · test {len(ids_te)}")
    print("La stratégie se choisit sur la validation. Le test n'est ouvert qu'ensuite.\n")

    print("Extraction des caractéristiques VGG16 (socle figé)")
    t0 = time.perf_counter()
    X_tr, _ = extraire(ids_tr)
    X_va, _ = extraire(ids_va)
    X_te, _ = extraire(ids_te)
    print(f"  {X_tr.shape[1]} dimensions · {time.perf_counter() - t0:.0f} s\n")

    strategies = [
        ("Sans augmentation", "aucune", 0),
        (f"Augmentation douce ×{copies}", "douce", copies),
        (f"Augmentation forte ×{copies}", "forte", copies),
        (f"Augmentation forte ×{2 * copies}", "forte", 2 * copies),
    ]

    # ------------------------------------------------ sélection, sur validation
    print("Comparaison sur les 157 produits de validation")
    selection, par_classe_val, modeles = [], {}, {}
    for nom, intensite, n in strategies:
        if n == 0:
            X, y, duree = X_tr, y_tr, 0.0
        else:
            t0 = time.perf_counter()
            X_aug, index = extraire(ids_tr, intensite=intensite, copies=n)
            X, y = np.vstack([X_tr, X_aug]), np.concatenate([y_tr, y_tr[index]])
            duree = time.perf_counter() - t0

        clf = tete().fit(X, y)
        pred_va = clf.predict(X_va)
        modeles[nom] = clf
        par_classe_val[nom] = _f1_par_classe(y_va, pred_va, etiquettes)
        selection.append(
            {
                "Stratégie": nom,
                "Images d'entraînement": int(X.shape[0]),
                "F1 macro (validation)": round(float(f1_score(y_va, pred_va, average="macro")), 4),
                "Exactitude (validation)": round(float(accuracy_score(y_va, pred_va)), 4),
            }
        )
        print(
            f"  {nom:28s} {X.shape[0]:5d} images · F1 macro "
            f"{selection[-1]['F1 macro (validation)']:.4f}" + (f" · {duree:.0f} s" if duree else "")
        )

    table_val = pd.DataFrame(selection).sort_values("F1 macro (validation)", ascending=False)
    table_val.to_csv(REPORTS / "supervise_image_validation.csv", index=False)
    pd.DataFrame(par_classe_val).to_csv(REPORTS / "supervise_image_par_classe_validation.csv")

    retenue = str(table_val.iloc[0]["Stratégie"])
    print(f"\n  Stratégie retenue : {retenue}")

    # ------------------------------------------------- une seule ouverture du test
    print("\nOuverture du jeu de test, avec la seule stratégie retenue")
    pred_te = modeles[retenue].predict(X_te)
    resultat = {
        "Stratégie retenue": retenue,
        "Exactitude (test)": round(float(accuracy_score(y_te, pred_te)), 4),
        "F1 macro (test)": round(float(f1_score(y_te, pred_te, average="macro")), 4),
    }
    par_classe_te = _f1_par_classe(y_te, pred_te, etiquettes)
    resultat["F1 classe la plus faible"] = min(par_classe_te.values())
    resultat["Classe la plus faible"] = min(par_classe_te, key=par_classe_te.get)
    pd.DataFrame([resultat]).to_csv(REPORTS / "supervise_image_test.csv", index=False)
    pd.Series(par_classe_te).to_csv(REPORTS / "supervise_image_par_classe_test.csv")
    print(
        f"  F1 macro {resultat['F1 macro (test)']:.4f} · "
        f"exactitude {resultat['Exactitude (test)']:.4f} · "
        f"{int(accuracy_score(y_te, pred_te) * len(y_te))}/{len(y_te)} bien classés"
    )

    # --------------------------------------------------------------- figures
    courts = [e.replace(" & ", "\n& ").replace(" and ", "\n& ") for e in etiquettes]

    fig, axe = plt.subplots(figsize=(7.5, 6.8))
    ConfusionMatrixDisplay.from_predictions(
        y_te,
        pred_te,
        display_labels=courts,
        ax=axe,
        colorbar=False,
        cmap="Blues",
        values_format="d",
    )
    axe.set_xlabel("Catégorie prédite")
    axe.set_ylabel("Catégorie réelle")
    axe.tick_params(labelsize=8)
    plt.setp(axe.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    axe.set_title(
        f"{retenue} : {len(y_te)} produits de test, ouverts une seule fois\n"
        f"F1 macro {resultat['F1 macro (test)']:.3f}".replace(".", ","),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "fig8_confusion_image.png", dpi=150)
    plt.close(fig)

    # L'augmentation ne dégrade pas uniformément : elle déplace les erreurs.
    # C'est ce que ce graphique montre, et il se lit sur la validation puisque
    # c'est là que la comparaison a eu lieu.
    tableau_pc = pd.DataFrame(par_classe_val)
    fig, axe = plt.subplots(figsize=(11, 5.5))
    largeur = 0.8 / len(tableau_pc.columns)
    x = np.arange(len(tableau_pc.index))
    for i, nom in enumerate(tableau_pc.columns):
        axe.bar(
            x + i * largeur - 0.4 + largeur / 2,
            tableau_pc[nom],
            width=largeur,
            label=nom,
            color=COULEURS[i % len(COULEURS)],
        )
    axe.set_xticks(x)
    axe.set_xticklabels(
        [e.replace(" & ", "\n& ").replace(" and ", "\n& ") for e in tableau_pc.index], fontsize=8
    )
    axe.set_ylabel("F1 sur la validation")
    axe.set_ylim(0, 1.05)
    axe.legend(fontsize=8.5, frameon=False, ncol=2)
    axe.set_title(
        "L'augmentation déplace les erreurs plutôt qu'elle ne les supprime : jeu de validation",
        fontsize=12,
    )
    for bord in ("top", "right"):
        axe.spines[bord].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig9_augmentation_par_classe.png", dpi=150)
    plt.close(fig)

    print("\n" + table_val.to_string(index=False))
    print("\nF1 par catégorie, sur la validation")
    print(tableau_pc.to_string())
    print("\nF1 par catégorie, sur le test (stratégie retenue)")
    print(pd.Series(par_classe_te).to_string())
    print(f"\n→ {REPORTS / 'supervise_image_test.csv'}, fig8 et fig9")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--copies", type=int, default=4, help="variantes augmentées par image")
    main(p.parse_args().copies)
