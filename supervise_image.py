"""Classification supervisée des images, avec et sans data augmentation.

python supervise_image.py              # les deux entraînements
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


def _mesurer(nom: str, y_vrai, y_pred, etiquettes: list[str]) -> dict:
    par_classe = f1_score(y_vrai, y_pred, average=None, labels=range(len(etiquettes)))
    return {
        "Modèle": nom,
        "Exactitude": round(float(accuracy_score(y_vrai, y_pred)), 4),
        "F1 macro": round(float(f1_score(y_vrai, y_pred, average="macro")), 4),
        "F1 classe la plus faible": round(float(par_classe.min()), 4),
        "Classe la plus faible": etiquettes[int(par_classe.argmin())],
        "_par_classe": dict(zip(etiquettes, [round(float(v), 3) for v in par_classe], strict=True)),
    }


def main(copies: int) -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load()
    train, val, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])
    etiquettes = list(enc.classes_)
    y_tr, y_te = enc.transform(train[LABEL_COL]), enc.transform(test[LABEL_COL])
    ids_tr, ids_te = train["uniq_id"].tolist(), test["uniq_id"].tolist()

    print(f"train {len(ids_tr)} · test {len(ids_te)} · {len(etiquettes)} catégories\n")

    print("Extraction des caractéristiques VGG16 (socle figé)")
    t0 = time.perf_counter()
    X_tr, _ = extraire(ids_tr)
    X_te, _ = extraire(ids_te)
    print(f"  {X_tr.shape[0]} images d'entraînement · {X_tr.shape[1]} dimensions")
    print(f"  {time.perf_counter() - t0:.0f} s\n")

    resultats = []

    print("Sans augmentation")
    clf = tete().fit(X_tr, y_tr)
    pred_sans = clf.predict(X_te)
    resultats.append(_mesurer("VGG16 figé + tête", y_te, pred_sans, etiquettes))
    print(
        f"  F1 macro {resultats[-1]['F1 macro']:.4f} · exactitude {resultats[-1]['Exactitude']:.4f}"
    )

    print(f"\nAvec augmentation — {copies} variantes par image")
    t0 = time.perf_counter()
    X_aug, index = extraire(ids_tr, augmenter=True, copies=copies)
    X_tr_aug = np.vstack([X_tr, X_aug])
    y_tr_aug = np.concatenate([y_tr, y_tr[index]])
    print(f"  {X_tr_aug.shape[0]} images après augmentation · {time.perf_counter() - t0:.0f} s")

    clf_aug = tete().fit(X_tr_aug, y_tr_aug)
    pred_avec = clf_aug.predict(X_te)
    resultats.append(_mesurer("VGG16 figé + tête, augmenté", y_te, pred_avec, etiquettes))
    print(
        f"  F1 macro {resultats[-1]['F1 macro']:.4f} · exactitude {resultats[-1]['Exactitude']:.4f}"
    )

    # --- sorties
    par_classe = {r["Modèle"]: r.pop("_par_classe") for r in resultats}
    tableau = pd.DataFrame(resultats)
    tableau.to_csv(REPORTS / "supervise_image.csv", index=False)
    pd.DataFrame(par_classe).to_csv(REPORTS / "supervise_image_par_classe.csv")

    courts = [e.replace(" & ", "\n& ").replace(" and ", "\n& ") for e in etiquettes]
    fig, (g, d) = plt.subplots(1, 2, figsize=(15, 6.5))
    for axe, pred, titre in (
        (g, pred_sans, "Sans augmentation"),
        (d, pred_avec, f"Avec augmentation ({copies} variantes)"),
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_te,
            pred,
            display_labels=courts,
            ax=axe,
            colorbar=False,
            cmap="Blues",
            values_format="d",
        )
        axe.set_title(titre, fontsize=11)
        axe.set_xlabel("Catégorie prédite")
        axe.set_ylabel("Catégorie réelle")
        axe.tick_params(labelsize=8)
    fig.suptitle(
        "Classification supervisée des images — 158 produits de test jamais vus", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "fig8_confusion_image.png", dpi=150)
    plt.close(fig)

    print("\n" + tableau.to_string(index=False))
    print("\nF1 par catégorie")
    print(pd.DataFrame(par_classe).to_string())
    print(f"\n→ {REPORTS / 'supervise_image.csv'} et fig8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--copies", type=int, default=4, help="variantes augmentées par image")
    main(p.parse_args().copies)
