"""Sérialise le modèle de la partie 4 — celui, et pas un autre.

Ce script ne choisit rien. La stratégie a été retenue sur la validation par
`supervise_image.py` ; on relit ce choix dans le rapport produit, on rejoue la
seule branche correspondante, et on vérifie que le modèle rechargé reproduit
le résultat publié avant de l'écrire sur disque.

Si la vérification échoue, le script s'arrête sans rien écrire. Il ne
réentraîne pas jusqu'à retrouver le chiffre attendu : un écart signifierait
que le protocole n'est pas reproductible, ce qui est l'information utile.

    python scripts/exporter_modele.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import LABEL_COL, load, split  # noqa: E402
from src.supervise_image import extraire, tete  # noqa: E402

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
ARTEFACT = MODELS / "image_vgg16_tete.joblib"
MONTRE = "08613e8b27838b997069b1fedb6e88d2"  # V9 METAL STRAP Analog Watch – For Men
TOLERANCE = 5e-4


def main() -> int:
    attendu = pd.read_csv(REPORTS / "supervise_image_test.csv").iloc[0]
    retenue = str(attendu["Stratégie retenue"])
    f1_attendu = float(attendu["F1 macro (test)"])
    exact_attendu = float(attendu["Exactitude (test)"])

    intensite, copies = retenue.split(" ")[1], int(retenue.rsplit("×", 1)[1])
    print(f"Stratégie retenue par la sélection sur validation : {retenue}")
    print(f"Résultat publié à reproduire : F1 macro {f1_attendu:.4f}\n")

    df = load()
    train, _, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])
    y_tr = enc.transform(train[LABEL_COL])
    y_te = enc.transform(test[LABEL_COL])
    ids_tr, ids_te = train["uniq_id"].tolist(), test["uniq_id"].tolist()

    print("Extraction VGG16 (socle figé) — base d'entraînement")
    X_tr, _ = extraire(ids_tr)
    print(f"  {X_tr.shape[0]} images · {X_tr.shape[1]} dimensions")

    print(f"Extraction des variantes augmentées — {intensite} ×{copies}")
    X_aug, index = extraire(ids_tr, intensite=intensite, copies=copies)
    X = np.vstack([X_tr, X_aug])
    y = np.concatenate([y_tr, y_tr[index]])
    print(f"  {X.shape[0]} images au total\n")

    clf = tete().fit(X, y)

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"tete": clf, "etiquettes": list(enc.classes_)}, ARTEFACT)

    # On vérifie le modèle tel qu'il sera chargé par l'application, pas celui
    # qui est encore en mémoire.
    recharge = joblib.load(ARTEFACT)
    tete_rechargee, etiquettes = recharge["tete"], recharge["etiquettes"]

    print("Vérification sur le jeu de test, modèle rechargé depuis le disque")
    X_te, _ = extraire(ids_te)
    pred = tete_rechargee.predict(X_te)
    f1 = float(f1_score(y_te, pred, average="macro"))
    exact = float(accuracy_score(y_te, pred))
    bons = int((pred == y_te).sum())
    print(f"  F1 macro {f1:.4f} (attendu {f1_attendu:.4f})")
    print(f"  Exactitude {exact:.4f} (attendu {exact_attendu:.4f})")
    print(f"  {bons}/{len(y_te)} bien classés\n")

    if abs(f1 - f1_attendu) > TOLERANCE or abs(exact - exact_attendu) > TOLERANCE:
        ARTEFACT.unlink(missing_ok=True)
        print("ÉCHEC — le protocole n'est pas reproduit. Artefact supprimé, rien n'est publié.")
        return 1

    rang = ids_te.index(MONTRE)
    probas = tete_rechargee.predict_proba(X_te[rang : rang + 1])[0]
    ordre = np.argsort(probas)[::-1]
    nom = df.loc[df["uniq_id"] == MONTRE, "product_name"].iloc[0].strip()
    print(f"Prédiction pour « {nom} » — jeu de test, jamais vu à l'entraînement")
    for i in ordre[:3]:
        print(f"  {etiquettes[i]:28s} {probas[i]:.4f}")
    print(f"\nCatégorie réelle : {df.loc[df['uniq_id'] == MONTRE, LABEL_COL].iloc[0]}")
    print(f"→ {ARTEFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
