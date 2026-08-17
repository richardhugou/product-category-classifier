"""Précision, rappel et F1 par catégorie, pour le modèle retenu, sur le jeu réservé.

Le projet rapportait le F1 macro et l'exactitude, jamais la précision ni le rappel
par catégorie. Ce script complète la description de la même évaluation : il
n'ouvre pas une seconde fois le jeu de test, il détaille l'unique ouverture déjà
faite pour le bras retenu.

Garde-fou : le total doit reproduire `reports/comparaison_test.csv`, sinon le
fichier de sortie n'est pas écrit.

    python scripts/rapport_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion import concatener, normaliser  # noqa: E402
from src.images import features  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402
from src.supervise_image import tete  # noqa: E402
from src.text import vectoriseur  # noqa: E402

SORTIE = ROOT / "reports" / "classification_test.csv"
ATTENDU = ROOT / "reports" / "comparaison_test.csv"
TOLERANCE = 5e-4


def main() -> int:
    attendu = pd.read_csv(ATTENDU).iloc[0]
    cible = float(attendu["F1 macro (test)"])

    df = load()
    train, _, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])
    y_tr, y_te = enc.transform(train[LABEL_COL]), enc.transform(test[LABEL_COL])

    vec = vectoriseur().fit(train[TEXT_COL])
    X_img, _ = features(df["uniq_id"])
    par_id = dict(zip(df["uniq_id"], X_img, strict=True))

    def X(part: pd.DataFrame):
        dino = np.vstack([par_id[u] for u in part["uniq_id"]])
        return concatener(vec.transform(part[TEXT_COL]), normaliser(dino))

    clf = tete().fit(X(train), y_tr)
    pred = clf.predict(X(test))

    f1m = float(f1_score(y_te, pred, average="macro"))
    if abs(f1m - cible) > TOLERANCE:
        print(f"ÉCHEC : F1 macro {f1m:.4f} au lieu de {cible:.4f}. Rien n'est écrit.")
        return 1

    rapport = classification_report(
        y_te, pred, target_names=list(enc.classes_), output_dict=True, zero_division=0
    )
    lignes = []
    for nom in list(enc.classes_) + ["macro avg", "weighted avg"]:
        d = rapport[nom]
        lignes.append(
            {
                "Catégorie": nom,
                "Précision": round(d["precision"], 3),
                "Rappel": round(d["recall"], 3),
                "F1": round(d["f1-score"], 3),
                "Articles": int(d["support"]),
            }
        )
    table = pd.DataFrame(lignes)
    table.to_csv(SORTIE, index=False)

    print(f"Modèle retenu : {attendu['Représentation retenue']}")
    print(f"Reproduit le résultat publié : F1 macro {f1m:.4f}\n")
    print(table.to_string(index=False))
    print(f"\nExactitude : {rapport['accuracy']:.4f}")
    print(f"→ {SORTIE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
