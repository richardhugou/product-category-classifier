"""Démonstration — catégorisation d'un article, texte seul ou texte + image.

    streamlit run app.py

Deux modèles tournent côte à côte : le modèle texte, quasi gratuit, et la
fusion texte + image, plus juste mais 600 fois plus coûteuse à l'inférence.
Le seuil de confiance matérialise UC4 — sous le seuil, le modèle ne tranche
pas et l'article part en revue humaine.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.images import image_path  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402

TEXTE_PATH = ROOT / "models" / "tfidf_mlp.joblib"
FUSION_PATH = ROOT / "models" / "fusion_mlp.joblib"
FEATURES = ROOT / "data" / "features"
SEUIL_DEFAUT = 0.60
MS_ENCODAGE_IMAGE = 35.7  # mesuré sur cette machine

st.set_page_config(page_title="Catégorisation d'articles", page_icon="🏷️", layout="wide")


def _empreinte(p: Path) -> str:
    """Version du modèle — REQ-08. Toute prédiction est rattachable à son modèle."""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


@st.cache_resource
def get_modeles():
    vec, clf = joblib.load(TEXTE_PATH)
    v_texte = _empreinte(TEXTE_PATH)
    fusion, v_fusion = None, None
    if FUSION_PATH.exists():
        _, fusion = joblib.load(FUSION_PATH)
        v_fusion = _empreinte(FUSION_PATH)
    return vec, clf, fusion, v_texte, v_fusion


@st.cache_resource
def get_encodeur_image():
    """Chargé seulement si une photographie est téléversée — 346 Mo."""
    import torch

    from src.images import _load_encoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return (*_load_encoder(device), device)


@st.cache_data
def get_exemples():
    df = load()
    _, _, test = split(df)
    return test.reset_index(drop=True)


@st.cache_data
def get_features_cache():
    x, ids = FEATURES / "dinov2_base.npy", FEATURES / "dinov2_base_ids.npy"
    if x.exists() and ids.exists():
        return dict(zip(np.load(ids, allow_pickle=True), np.load(x), strict=True))
    return {}


def vecteur_image(uniq_id=None, fichier=None):
    """Renvoie (vecteur 1536, millisecondes). Sert le cache si l'article est connu."""
    cache = get_features_cache()
    if fichier is None and uniq_id is not None and uniq_id in cache:
        return cache[uniq_id], MS_ENCODAGE_IMAGE  # coût réel réintégré
    if fichier is None:
        return None, 0.0

    import torch
    from PIL import Image

    proc, model, device = get_encodeur_image()
    image = Image.open(fichier).convert("RGB")
    t0 = time.perf_counter()
    with torch.no_grad():
        entrees = proc(images=[image], return_tensors="pt").to(device)
        h = model(**entrees).last_hidden_state
        v = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1).cpu().numpy()[0]
    return v, (time.perf_counter() - t0) * 1000


vec, clf, fusion, v_texte, v_fusion = get_modeles()
test = get_exemples()
labels = sorted(test[LABEL_COL].unique())

st.title("Catégorisation automatique d'articles")
st.caption(
    "7 catégories · entraîné sur 735 articles, évalué sur 158 jamais vus · "
    "seuil métier F1 macro ≥ 0,90"
)

# ---------------------------------------------------------------- barre latérale
seuil = st.sidebar.slider(
    "Seuil de confiance",
    0.0,
    1.0,
    SEUIL_DEFAUT,
    0.05,
    help="En dessous, l'article part en revue humaine plutôt que d'être classé à tort.",
)
st.sidebar.markdown("---")
st.sidebar.subheader("Charger un exemple du jeu de test")
idx = st.sidebar.number_input("Index", 0, len(test) - 1, 0, 1)
if st.sidebar.button("Charger", use_container_width=True):
    ligne = test.loc[idx]
    st.session_state.update(texte=ligne[TEXT_COL], vrai=ligne[LABEL_COL], uniq=ligne["uniq_id"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Modèle texte `{v_texte}`")
if v_fusion:
    st.sidebar.caption(f"Modèle fusion `{v_fusion}`")

# ---------------------------------------------------------------- saisie
gauche, droite = st.columns([3, 2])

with gauche:
    texte = st.text_area(
        "Description de l'article",
        value=st.session_state.get("texte", ""),
        height=190,
        placeholder="Coller ici la description d'un produit, en anglais…",
    )

with droite:
    televerse = st.file_uploader("Photographie de l'article", type=["jpg", "jpeg", "png"])
    uniq = st.session_state.get("uniq")
    chemin = image_path(uniq) if uniq else None
    apercu = televerse or (chemin if chemin and chemin.exists() else None)
    if apercu is not None:
        st.image(
            apercu,
            use_container_width=True,
            caption="Photographie téléversée" if televerse else "Photographie de l'article chargé",
        )
    else:
        st.info("Sans photographie, seul le modèle texte se prononce.")

# ---------------------------------------------------------------- prédiction
if st.button("Catégoriser", type="primary") and texte.strip():
    t0 = time.perf_counter()
    X = vec.transform([texte])
    p_texte = clf.predict_proba(X)[0]
    ms_texte = (time.perf_counter() - t0) * 1000

    v_img, ms_img = vecteur_image(uniq_id=None if televerse else uniq, fichier=televerse)
    p_fusion = ms_fusion = None
    if fusion is not None and v_img is not None:
        from scipy.sparse import csr_matrix, hstack
        from sklearn.preprocessing import normalize

        t0 = time.perf_counter()
        F = hstack([X, csr_matrix(normalize(v_img.reshape(1, -1)))]).tocsr()
        p_fusion = fusion.predict_proba(F)[0]
        ms_fusion = (time.perf_counter() - t0) * 1000 + ms_img

    def bloc(titre, proba, ms, version):
        i = int(proba.argmax())
        cat, conf = labels[i], float(proba[i])
        st.markdown(f"**{titre}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Catégorie", cat if conf >= seuil else "En revue")
        c2.metric("Confiance", f"{conf:.1%}")
        c3.metric("Inférence", f"{ms:.1f} ms")
        if conf < seuil:
            st.warning(
                f"Confiance sous {seuil:.0%} : l'article est mis en attente plutôt que "
                f"classé à tort. La proposition serait « {cat} »."
            )
        else:
            st.success(f"Suggestion : **{cat}** — modifiable par le vendeur.")
        st.caption(f"Modèle `{version}`")
        return cat

    ca, cb = st.columns(2)
    with ca:
        cat_texte = bloc("Texte seul", p_texte, ms_texte, v_texte)
    with cb:
        if p_fusion is not None:
            cat_fusion = bloc("Texte + image", p_fusion, ms_fusion, v_fusion)
        else:
            st.markdown("**Texte + image**")
            st.info("Fournir une photographie pour activer la fusion.")
            cat_fusion = None

    if st.session_state.get("texte") == texte and "vrai" in st.session_state:
        vrai = st.session_state["vrai"]
        parts = [
            f"Catégorie réelle : **{vrai}**",
            "texte " + ("correct" if cat_texte == vrai else "erroné"),
        ]
        if cat_fusion is not None:
            parts.append("fusion " + ("correcte" if cat_fusion == vrai else "erronée"))
        st.caption(" — ".join(parts))

    st.subheader("Distribution des probabilités")
    d = {"texte seul": p_texte}
    if p_fusion is not None:
        d["texte + image"] = p_fusion
    st.bar_chart(pd.DataFrame(d, index=labels))

# ---------------------------------------------------------------- benchmark
with st.expander("Le benchmark"):
    p = ROOT / "reports" / "benchmark.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        st.caption(
            "Cinq modèles sur six franchissent le seuil métier : il ne départage donc rien. "
            "Le critère de décision devient le coût. La fusion gagne 3 points de F1 — et "
            "surtout 8 points sur la catégorie la plus faible — pour 600 fois le coût "
            "d'inférence du modèle texte."
        )
    else:
        st.info("Lancer `python benchmark.py --encoders --images` pour produire le tableau.")
