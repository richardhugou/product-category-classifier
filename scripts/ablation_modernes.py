"""Pourquoi ModernBERT ne fait-il pas mieux ? Ablation, sur validation seulement.

Le résultat à expliquer : figés et moyennés, BERT (0,905) et ModernBERT (0,904)
restent derrière TF-IDF (0,9365), quand DINOv2 domine largement VGG16 côté
image. Quatre hypothèses mesurables :

    H1 — le regroupement : la moyenne des jetons dilue les mots rares ;
         le jeton de classe s'en sortirait-il mieux ?
    H2 — la troncature : 256 jetons coupent les fiches longues ; ModernBERT
         accepte 8 192 — que donne 1 024 ?
    H3 — l'échelle : les vecteurs non normalisés désavantagent-ils la tête ?
    H4 — côté image : DINOv2 gagne-t-il par son architecture (Vision
         Transformer) ou par son pré-entraînement auto-supervisé ? Un ViT
         supervisé ImageNet départage.

Et deux diagnostics : la part de fiches réellement tronquées, et la similarité
moyenne entre fiches — si le gabarit commun écrase tout, deux fiches au hasard
se ressemblent déjà beaucoup pour un encodeur contextuel.

Le jeu de test n'est jamais touché : on cherche à comprendre, pas à conclure.

    python scripts/ablation_modernes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.fusion import normaliser  # noqa: E402
from src.images import image_path  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402
from src.supervise_image import tete  # noqa: E402
from src.text import ENCODEURS, charger_encodeur, vectoriseur  # noqa: E402

REPORTS = ROOT / "reports"
VIT_SUPERVISE = "google/vit-base-patch16-224"


def encoder_variante(textes, tok, mdl, device, regroupement="moyenne", max_len=256):
    """Encode avec le regroupement et la longueur demandés."""
    import torch

    sorties = []
    with torch.no_grad():
        for i in range(0, len(textes), 32):
            lot = tok(
                textes[i : i + 32],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            h = mdl(**lot).last_hidden_state
            if regroupement == "cls":
                sorties.append(h[:, 0].cpu().numpy())
            else:
                masque = lot["attention_mask"].unsqueeze(-1).float()
                sorties.append(((h * masque).sum(1) / masque.sum(1)).cpu().numpy())
    return np.vstack(sorties)


def encoder_image(uniq_ids, model_id, device):
    """Jeton de classe + moyenne des patches — le même protocole que DINOv2."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id).to(device).eval()
    sorties = []
    with torch.no_grad():
        for i in range(0, len(uniq_ids), 16):
            lot = [Image.open(image_path(u)).convert("RGB") for u in uniq_ids[i : i + 16]]
            entrees = proc(images=lot, return_tensors="pt").to(device)
            h = mdl(**entrees).last_hidden_state
            sorties.append(np.hstack([h[:, 0].cpu().numpy(), h[:, 1:].mean(1).cpu().numpy()]))
    return np.vstack(sorties)


def main() -> None:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    df = load()
    train, val, _ = split(df)
    from sklearn.preprocessing import LabelEncoder

    enc = LabelEncoder().fit(df[LABEL_COL])
    y_tr, y_va = enc.transform(train[LABEL_COL]), enc.transform(val[LABEL_COL])
    txt_tr, txt_va = train[TEXT_COL].tolist(), val[TEXT_COL].tolist()

    def mesurer(X_tr, X_va) -> float:
        pred = tete().fit(X_tr, y_tr).predict(X_va)
        return round(float(f1_score(y_va, pred, average="macro")), 4)

    lignes = []

    # ------------------------------------------------- texte : H1, H2, H3
    for nom, model_id in ENCODEURS.items():
        tok, mdl, _ = charger_encodeur(model_id, device)
        variantes = [("moyenne · 256", "moyenne", 256), ("cls · 256", "cls", 256)]
        if "Modern" in nom:
            variantes.append(("moyenne · 1024", "moyenne", 1024))
        for lib, regroupement, max_len in variantes:
            t0 = time.perf_counter()
            E_tr = encoder_variante(txt_tr, tok, mdl, device, regroupement, max_len)
            E_va = encoder_variante(txt_va, tok, mdl, device, regroupement, max_len)
            f1_brut = mesurer(E_tr, E_va)
            f1_l2 = mesurer(normalize(E_tr), normalize(E_va))
            lignes.append(
                {"Bras": f"{nom} · {lib}", "F1 macro (val)": f1_brut, "F1 macro (val, L2)": f1_l2}
            )
            print(
                f"  {nom} · {lib:14s} brut {f1_brut:.4f} · L2 {f1_l2:.4f}"
                f" · {time.perf_counter() - t0:.0f} s"
            )
        del tok, mdl

    # ------------------------------------------------- image : H4
    print("ViT supervisé ImageNet (même protocole d'extraction que DINOv2)")
    ids_tr, ids_va = train["uniq_id"].tolist(), val["uniq_id"].tolist()
    t0 = time.perf_counter()
    V_tr = normaliser(encoder_image(ids_tr, VIT_SUPERVISE, device))
    V_va = normaliser(encoder_image(ids_va, VIT_SUPERVISE, device))
    f1_vit = mesurer(V_tr, V_va)
    lignes.append(
        {
            "Bras": "ViT-B/16 supervisé ImageNet",
            "F1 macro (val)": f1_vit,
            "F1 macro (val, L2)": f1_vit,
        }
    )
    print(f"  F1 macro {f1_vit:.4f} · {time.perf_counter() - t0:.0f} s")

    table = pd.DataFrame(lignes)
    table.to_csv(REPORTS / "ablation_validation.csv", index=False)

    # ------------------------------------------------- diagnostics
    print("\nDiagnostics")
    tok, _, _ = charger_encodeur(ENCODEURS["BERT figé (2018)"], "cpu")
    longueurs = [len(tok(t, truncation=False)["input_ids"]) for t in txt_tr + txt_va]
    tronquees = sum(1 for n in longueurs if n > 256)
    print(
        f"  fiches au-delà de 256 jetons : {tronquees}/{len(longueurs)}"
        f" ({100 * tronquees / len(longueurs):.1f} %) · max {max(longueurs)}"
    )

    tok, mdl, _ = charger_encodeur(ENCODEURS["BERT figé (2018)"], device)
    E_va = normalize(encoder_variante(txt_va, tok, mdl, device))
    sim_bert = (E_va @ E_va.T)[np.triu_indices(len(E_va), k=1)].mean()
    vec = vectoriseur().fit(txt_tr)
    T_va = vec.transform(txt_va)
    sim_tfidf = np.asarray((T_va @ T_va.T).todense())[np.triu_indices(T_va.shape[0], k=1)].mean()
    print(
        f"  similarité cosinus moyenne entre deux fiches de validation :"
        f" BERT {sim_bert:.3f} · TF-IDF {sim_tfidf:.3f}"
    )

    # Les fiches que BERT rate et que TF-IDF classe bien.
    E_tr = encoder_variante(txt_tr, tok, mdl, device)
    pred_bert = tete().fit(E_tr, y_tr).predict(encoder_variante(txt_va, tok, mdl, device))
    pred_tfidf = tete().fit(vec.transform(txt_tr), y_tr).predict(T_va)
    rattrapees = [
        (i, val.iloc[i])
        for i in range(len(y_va))
        if pred_bert[i] != y_va[i] and pred_tfidf[i] == y_va[i]
    ]
    print(f"  fiches ratées par BERT et rattrapées par TF-IDF : {len(rattrapees)}")
    for i, ligne in rattrapees[:3]:
        print(
            f"    « {str(ligne['product_name']).strip()[:48]} » — "
            f"{enc.classes_[y_va[i]]} lu {enc.classes_[pred_bert[i]]}"
        )

    print("\n" + table.to_string(index=False))
    print(f"\n→ {REPORTS / 'ablation_validation.csv'}")


if __name__ == "__main__":
    main()
