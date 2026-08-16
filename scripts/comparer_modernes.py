"""Références contre modernes : six représentations, une seule discipline.

Il s'agit de comparer les solutions « historiques » du projet à des options
plus récentes, sous le protocole de la partie 4 : même découpe stratifiée,
même tête de classification, comparaison sur les 157 produits de validation —
et le test n'est ouvert qu'une fois, pour le seul bras retenu.

    Texte — TF-IDF (référence) · BERT figé (2018) · ModernBERT figé (2024)
    Image — VGG16 figé (référence CNN) · DINOv2 figé (Vision Transformer)
    Mixte — description TF-IDF + image DINOv2, concaténées

Aucune augmentation ici : on compare des représentations, pas des stratégies
d'entraînement. Les encodeurs restent figés — aucun poids n'est réglé.

    python scripts/comparer_modernes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion import concatener, normaliser  # noqa: E402
from src.images import features  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402
from src.supervise_image import extraire, tete  # noqa: E402
from src.text import ENCODEURS, charger_encodeur, encoder, vectoriseur  # noqa: E402

REPORTS = ROOT / "reports"

# Références en gris, modernes en bleu, le mixte en terre cuite.
TEINTES = {"référence": "#98A3AD", "moderne": "#0E3A5C", "mixte": "#A0430F"}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df = load()
    train, val, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])
    etiquettes = list(enc.classes_)
    y_tr, y_va, y_te = (enc.transform(d[LABEL_COL]) for d in (train, val, test))
    txt = {n: d[TEXT_COL].tolist() for n, d in (("tr", train), ("va", val), ("te", test))}
    ids = {n: d["uniq_id"].tolist() for n, d in (("tr", train), ("va", val), ("te", test))}

    print(f"train {len(y_tr)} · validation {len(y_va)} · test {len(y_te)}")
    print("Comparaison sur la validation. Le test n'est ouvert qu'une fois, à la fin.\n")

    bras: dict[str, dict] = {}

    # ------------------------------------------------------------- texte
    print("TF-IDF (référence texte)")
    vec = vectoriseur()
    bras["TF-IDF"] = {
        "famille": "texte",
        "statut": "référence",
        "tr": vec.fit_transform(txt["tr"]),
        "va": vec.transform(txt["va"]),
        "te": vec.transform(txt["te"]),
    }

    for nom, model_id in ENCODEURS.items():
        print(nom)
        t0 = time.perf_counter()
        tok, mdl, device = charger_encodeur(model_id)
        jeux = {n: encoder(txt[n], tok, mdl, device) for n in ("tr", "va", "te")}
        del tok, mdl
        print(f"  encodé en {time.perf_counter() - t0:.0f} s")
        statut = "moderne" if "Modern" in nom else "référence"
        bras[nom] = {"famille": "texte", "statut": statut, **jeux}

    # ------------------------------------------------------------- image
    print("VGG16 figé (référence CNN)")
    t0 = time.perf_counter()
    vgg = {n: extraire(ids[n])[0] for n in ("tr", "va", "te")}
    print(f"  extrait en {time.perf_counter() - t0:.0f} s")
    bras["VGG16 figé (CNN)"] = {"famille": "image", "statut": "référence", **vgg}

    print("DINOv2 figé (Vision Transformer)")
    X_img, _ = features(df["uniq_id"])
    par_id = dict(zip(df["uniq_id"], X_img, strict=True))
    dino = {n: normaliser(np.vstack([par_id[u] for u in ids[n]])) for n in ("tr", "va", "te")}
    bras["DINOv2 figé (ViT)"] = {"famille": "image", "statut": "moderne", **dino}

    # ------------------------------------------------------------- mixte
    bras["Fusion TF-IDF + DINOv2"] = {
        "famille": "mixte",
        "statut": "mixte",
        "tr": concatener(bras["TF-IDF"]["tr"], dino["tr"]),
        "va": concatener(bras["TF-IDF"]["va"], dino["va"]),
        "te": concatener(bras["TF-IDF"]["te"], dino["te"]),
    }

    # ------------------------------------- même tête partout, lecture sur validation
    print("\nMême tête de classification pour les six bras — validation")
    lignes, par_classe, modeles = [], {}, {}
    for nom, b in bras.items():
        clf = tete().fit(b["tr"], y_tr)
        pred = clf.predict(b["va"])
        scores = f1_score(y_va, pred, average=None, labels=range(len(etiquettes)))
        par_classe[nom] = dict(zip(etiquettes, [round(float(v), 3) for v in scores], strict=True))
        modeles[nom] = clf
        lignes.append(
            {
                "Représentation": nom,
                "Famille": b["famille"],
                "Statut": b["statut"],
                "Dimensions": int(b["tr"].shape[1]),
                "F1 macro (validation)": round(float(f1_score(y_va, pred, average="macro")), 4),
                "Exactitude (validation)": round(float(accuracy_score(y_va, pred)), 4),
                "F1 classe la plus faible": round(float(scores.min()), 3),
                "Classe la plus faible": etiquettes[int(scores.argmin())],
            }
        )
        print(
            f"  {nom:26s} F1 macro {lignes[-1]['F1 macro (validation)']:.4f}"
            f" · min {lignes[-1]['F1 classe la plus faible']:.3f}"
        )

    table = pd.DataFrame(lignes).sort_values("F1 macro (validation)", ascending=False)
    table.to_csv(REPORTS / "comparaison_validation.csv", index=False)
    pd.DataFrame(par_classe).to_csv(REPORTS / "comparaison_par_classe_validation.csv")

    # ------------------------------------------------- une seule ouverture du test
    retenue = str(table.iloc[0]["Représentation"])
    print(f"\nBras retenu sur validation : {retenue}")
    pred_te = modeles[retenue].predict(bras[retenue]["te"])
    scores_te = f1_score(y_te, pred_te, average=None, labels=range(len(etiquettes)))
    resultat = {
        "Représentation retenue": retenue,
        "F1 macro (test)": round(float(f1_score(y_te, pred_te, average="macro")), 4),
        "Exactitude (test)": round(float(accuracy_score(y_te, pred_te)), 4),
        "Bien classés": f"{int((pred_te == y_te).sum())}/{len(y_te)}",
        "F1 classe la plus faible (test)": round(float(scores_te.min()), 3),
        "Classe la plus faible (test)": etiquettes[int(scores_te.argmin())],
    }
    pd.DataFrame([resultat]).to_csv(REPORTS / "comparaison_test.csv", index=False)
    pd.Series(dict(zip(etiquettes, [round(float(v), 3) for v in scores_te], strict=True))).to_csv(
        REPORTS / "comparaison_par_classe_test.csv"
    )
    print(
        f"  F1 macro test {resultat['F1 macro (test)']:.4f}"
        f" · {resultat['Bien classés']} bien classés"
    )

    # ------------------------------------------------------------- figure
    ordre = list(table["Représentation"])
    courts = [e.replace(" & ", "\n& ").replace(" and ", "\n& ") for e in etiquettes]
    fig, axe = plt.subplots(figsize=(12.5, 5.8))
    largeur = 0.8 / len(ordre)
    x = np.arange(len(etiquettes))
    for i, nom in enumerate(ordre):
        statut = str(table.set_index("Représentation").loc[nom, "Statut"])
        axe.bar(
            x + i * largeur - 0.4 + largeur / 2,
            [par_classe[nom][e] for e in etiquettes],
            width=largeur,
            label=nom,
            color=TEINTES[statut],
            alpha=1.0 if statut != "référence" else 0.85,
        )
    axe.set_xticks(x)
    axe.set_xticklabels(courts, fontsize=8)
    axe.set_ylabel("F1 sur la validation")
    axe.set_ylim(0, 1.05)
    axe.legend(fontsize=8, frameon=False, ncol=2)
    axe.set_title(
        "Références et modernes, catégorie par catégorie — jeu de validation", fontsize=12
    )
    for bord in ("top", "right"):
        axe.spines[bord].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig10_comparaison_par_classe.png", dpi=150)

    print("\n" + table.to_string(index=False))
    print(f"\n→ {REPORTS / 'comparaison_validation.csv'}, comparaison_test.csv, fig10")


if __name__ == "__main__":
    main()
