"""Sérialise la solution retenue — fusion TF-IDF + DINOv2 — et la vérifie.

Le bras retenu vient de `reports/comparaison_validation.csv` : ce script ne
choisit rien. Il rejoue l'entraînement de ce seul bras, vérifie que le modèle
rechargé depuis le disque reproduit `reports/comparaison_test.csv`, puis donne
la prédiction de la montre V9 et les erreurs restantes du jeu de test.

Si la vérification échoue, l'artefact est supprimé et rien n'est publié.

    python scripts/exporter_fusion.py
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
from src.fusion import concatener, normaliser  # noqa: E402
from src.images import features  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402
from src.supervise_image import tete  # noqa: E402
from src.text import vectoriseur  # noqa: E402

REPORTS = ROOT / "reports"
ARTEFACT = ROOT / "models" / "fusion_tfidf_dinov2.joblib"
MONTRE = "08613e8b27838b997069b1fedb6e88d2"
TOLERANCE = 5e-4


def main() -> int:
    attendu = pd.read_csv(REPORTS / "comparaison_test.csv").iloc[0]
    retenue = str(attendu["Représentation retenue"])
    if "TF-IDF + DINOv2" not in retenue:
        print(f"Bras retenu inattendu : {retenue} — ce script ne sait sérialiser que la fusion.")
        return 1
    print(f"Bras retenu sur validation : {retenue}")
    print(f"À reproduire : F1 macro {attendu['F1 macro (test)']:.4f} · {attendu['Bien classés']}\n")

    df = load()
    train, _, test = split(df)
    enc = LabelEncoder().fit(df[LABEL_COL])
    y_tr, y_te = enc.transform(train[LABEL_COL]), enc.transform(test[LABEL_COL])

    vec = vectoriseur().fit(train[TEXT_COL])
    X_img, _ = features(df["uniq_id"])
    par_id = dict(zip(df["uniq_id"], X_img, strict=True))
    dino = {
        n: np.vstack([par_id[u] for u in d["uniq_id"]]) for n, d in (("tr", train), ("te", test))
    }
    X_tr = concatener(vec.transform(train[TEXT_COL]), normaliser(dino["tr"]))
    clf = tete().fit(X_tr, y_tr)

    joblib.dump({"vectoriseur": vec, "tete": clf, "etiquettes": list(enc.classes_)}, ARTEFACT)

    # Le modèle tel qu'il sera chargé, pas celui encore en mémoire.
    paquet = joblib.load(ARTEFACT)
    X_te = concatener(paquet["vectoriseur"].transform(test[TEXT_COL]), normaliser(dino["te"]))
    pred = paquet["tete"].predict(X_te)
    f1 = float(f1_score(y_te, pred, average="macro"))
    print("Vérification, modèle rechargé depuis le disque")
    print(f"  exactitude {accuracy_score(y_te, pred):.4f}")
    print(f"  F1 macro {f1:.4f} (attendu {attendu['F1 macro (test)']:.4f})")
    print(f"  {int((pred == y_te).sum())}/{len(y_te)} bien classés")

    if abs(f1 - float(attendu["F1 macro (test)"])) > TOLERANCE:
        ARTEFACT.unlink(missing_ok=True)
        print("ÉCHEC — le protocole n'est pas reproduit. Artefact supprimé.")
        return 1

    ids_te = test["uniq_id"].tolist()
    rang = ids_te.index(MONTRE)
    probas = paquet["tete"].predict_proba(X_te[rang])[0]
    ordre = np.argsort(probas)[::-1]
    print("\nPrédiction pour la montre V9 — jeu de test")
    for i in ordre[:3]:
        print(f"  {paquet['etiquettes'][i]:28s} {probas[i]:.4f}")

    print("\nLes erreurs restantes du jeu de test")
    for i in np.flatnonzero(pred != y_te):
        nom = str(test.iloc[i]["product_name"]).strip()[:52]
        print(f"  « {nom} » — {enc.classes_[y_te[i]]} lu {enc.classes_[pred[i]]}")
    print(f"\n→ {ARTEFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
