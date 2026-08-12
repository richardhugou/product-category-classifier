"""Comparaison des approches, à protocole constant.

Le seuil métier est fixé avant toute mesure (`src.evaluate.SEUIL_F1_METIER`).
Les axes rapportés ne sont pas seulement la performance : le temps
d'entraînement, le temps d'inférence **de bout en bout** et l'empreinte
déployée décident autant, et souvent plus.

    python benchmark.py             # texte classique — socle, sans torch
    python benchmark.py --encoders  # + BERT et ModernBERT figés
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from src.evaluate import SEUIL_F1_METIER, mesurer  # noqa: E402
from src.pipeline import LABEL_COL, SEED, TEXT_COL, load, split  # noqa: E402
from src.text import vectoriseur  # noqa: E402

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

def _mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=SEED
    )


def _chronometrer(fn):
    t0 = time.perf_counter()
    resultat = fn()
    return resultat, time.perf_counter() - t0


def main(avec_encodeurs: bool) -> None:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = load()
    train, val, test = split(df)
    encodeur_etiquettes = LabelEncoder().fit(df[LABEL_COL])
    etiquettes = list(encodeur_etiquettes.classes_)
    y_tr = encodeur_etiquettes.transform(train[LABEL_COL])
    y_te = encodeur_etiquettes.transform(test[LABEL_COL])
    txt_tr, txt_te = train[TEXT_COL].tolist(), test[TEXT_COL].tolist()
    n_te = len(y_te)

    print(f"train {len(train)} · val {len(val)} · test {n_te} · {len(etiquettes)} classes")
    print(f"seuil métier : F1 macro >= {SEUIL_F1_METIER:.2f}, fixé avant les mesures\n")

    lignes: list[dict] = []

    def ajouter(nom, y_pred, entrainement_s, inference_ms, empreinte_mo):
        ligne = mesurer(nom, y_te, y_pred, etiquettes, entrainement_s, inference_ms, empreinte_mo)
        lignes.append(ligne)
        print(
            f"  {nom:30s} F1 {ligne['F1 macro']:.4f} · min {ligne['F1 classe min']:.3f}"
            f" · {entrainement_s:6.2f} s · {inference_ms:6.2f} ms · {empreinte_mo:6.1f} Mo"
        )

    # ---------------------------------------------------------------- TF-IDF
    print("TF-IDF")
    vec = vectoriseur()
    Xtr, vec_s = _chronometrer(lambda: vec.fit_transform(txt_tr))
    print(f"  vocabulaire {Xtr.shape[1]} termes, vectorisation {vec_s:.2f} s")

    import xgboost as xgb

    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    _, t = _chronometrer(lambda: xgb_clf.fit(Xtr, y_tr))
    joblib.dump((vec, xgb_clf), MODELS / "tfidf_xgb.joblib")
    # Inférence de bout en bout : la vectorisation est dans le chronomètre.
    pred, t_inf = _chronometrer(lambda: xgb_clf.predict(vec.transform(txt_te)))
    ajouter(
        "TF-IDF + XGBoost",
        pred,
        t + vec_s,
        t_inf * 1000 / n_te,
        (MODELS / "tfidf_xgb.joblib").stat().st_size / 1e6,
    )

    mlp = _mlp()
    _, t = _chronometrer(lambda: mlp.fit(Xtr, y_tr))
    joblib.dump((vec, mlp), MODELS / "tfidf_mlp.joblib")
    pred, t_inf = _chronometrer(lambda: mlp.predict(vec.transform(txt_te)))
    ajouter(
        "TF-IDF + MLP",
        pred,
        t + vec_s,
        t_inf * 1000 / n_te,
        (MODELS / "tfidf_mlp.joblib").stat().st_size / 1e6,
    )

    # ------------------------------------------------- encodeurs de texte figés
    if avec_encodeurs:
        from sklearn.linear_model import LogisticRegression

        from src.text import ENCODEURS, charger_encodeur, empreinte_mo, encoder

        print("\nEncodeurs de texte figés — même tête, même découpe")
        for nom, model_id in ENCODEURS.items():
            try:
                # Le chargement du modèle est un coût de démarrage, pas d'inférence.
                tok, mdl, device = charger_encodeur(model_id)
                mo = empreinte_mo(mdl)

                def entrainer(tok=tok, mdl=mdl, device=device):
                    E = encoder(txt_tr, tok, mdl, device)
                    tete = LogisticRegression(max_iter=2000, random_state=SEED)
                    tete.fit(E, y_tr)
                    return tete

                tete, t = _chronometrer(entrainer)
                pred, t_inf = _chronometrer(
                    lambda tete=tete, tok=tok, mdl=mdl, device=device: tete.predict(
                        encoder(txt_te, tok, mdl, device)
                    )
                )
                ajouter(nom, pred, t, t_inf * 1000 / n_te, mo)
            except Exception as e:  # réseau, poids indisponibles, mémoire
                print(f"  {nom:30s} ÉCHEC — {type(e).__name__}: {str(e)[:80]}")

    # ------------------------------------------------------------- sorties
    par_classe = {ligne["Modèle"]: ligne.pop("_par_classe") for ligne in lignes}
    tableau = pd.DataFrame(lignes).sort_values("F1 macro", ascending=False)
    tableau.to_csv(REPORTS / "benchmark.csv", index=False)
    (REPORTS / "f1_par_classe.json").write_text(
        json.dumps(par_classe, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + tableau.to_string(index=False))
    print(f"\n→ {REPORTS / 'benchmark.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoders", action="store_true", help="BERT et ModernBERT figés")
    a = p.parse_args()
    main(a.encoders)
