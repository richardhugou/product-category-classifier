"""Démonstration — le modèle de la partie 4, et lui seul.

    streamlit run demo.py

Un seul chemin : la photographie entre, VGG16 figé en tire 512 nombres, la
tête rend 7 probabilités. C'est exactement le modèle sélectionné sur la
validation puis évalué sur le jeu réservé — F1 macro 0,867, 137 produits sur
158. Aucun autre modèle n'est chargé ici : `app.py` compare le texte, l'image
et leur fusion, ce qui relève d'un travail complémentaire et d'un autre
protocole.

Le modèle est produit par `scripts/exporter_modele.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.pipeline import LABEL_COL, load, split  # noqa: E402
from src.supervise_image import _appareil, _socle  # noqa: E402

ARTEFACT = ROOT / "models" / "image_vgg16_tete.joblib"
MONTRE = "08613e8b27838b997069b1fedb6e88d2"
ACC = "#14486B"

st.set_page_config(page_title="Catégoriser un article", page_icon="🏷️", layout="centered")


@st.cache_resource
def charger():
    if not ARTEFACT.exists():
        st.error("Modèle absent. Lancer d'abord `python scripts/exporter_modele.py`.")
        st.stop()
    paquet = joblib.load(ARTEFACT)
    device = _appareil()
    return paquet["tete"], paquet["etiquettes"], device, _socle(device)


@st.cache_data
def catalogue():
    """Les 158 produits du jeu réservé — les seuls que le modèle n'a jamais vus."""
    from src.pretraitement import chemin_image

    _, _, test = split(load())
    test = test.copy()
    test["chemin"] = test["uniq_id"].map(lambda u: str(chemin_image(u)))
    return test


def encoder(image: Image.Image, device, socle) -> np.ndarray:
    """Le même prétraitement qu'à l'entraînement — sinon le modèle voit autre chose."""
    import torch
    from torchvision.transforms import functional as F

    from src.supervise_image import transformations

    reseau, moyenne, ecart = socle
    x = F.to_tensor(transformations("aucune")(image.convert("RGB")))[None].to(device)
    x = (x - moyenne) / ecart
    with torch.no_grad():
        return reseau(x).mean(dim=(2, 3)).cpu().numpy()


tete, etiquettes, device, socle = charger()

st.title("Catégoriser un article")
st.caption(
    "Photographie → VGG16 figé → 512 caractéristiques → tête de classification → 7 catégories. "
    "Modèle évalué sur 158 produits réservés : F1 macro 0,867."
)

test = catalogue()
onglet_cat, onglet_fichier = st.tabs(["Un produit du catalogue", "Ma propre image"])

with onglet_cat:
    noms = test["product_name"].str.strip().tolist()
    defaut = test.index.get_indexer([test.index[test["uniq_id"] == MONTRE][0]])[0]
    choix = st.selectbox(
        "Article", range(len(noms)), index=int(defaut), format_func=lambda i: noms[i]
    )
    ligne = test.iloc[choix]
    image, verite = Image.open(ligne["chemin"]), ligne[LABEL_COL]

with onglet_fichier:
    depose = st.file_uploader("Une photographie de produit", type=["jpg", "jpeg", "png"])
    if depose is not None:
        image, verite = Image.open(depose), None

gauche, droite = st.columns([1, 1.35], gap="large")
with gauche:
    st.image(image, width="stretch")

probas = tete.predict_proba(encoder(image, device, socle))[0]
ordre = np.argsort(probas)[::-1]

with droite:
    st.markdown(
        f"<div style='font-size:2.6rem;line-height:1.1;color:{ACC};font-weight:600'>"
        f"{etiquettes[ordre[0]]}</div>"
        f"<div style='font-size:1.5rem;color:#4A525A'>confiance {probas[ordre[0]]:.3f}</div>".replace(
            f"{probas[ordre[0]]:.3f}", f"{probas[ordre[0]]:.3f}".replace(".", ",")
        ),
        unsafe_allow_html=True,
    )
    if verite is not None:
        juste = etiquettes[ordre[0]] == verite
        st.caption(f"Catégorie réelle : **{verite}** — {'correct' if juste else 'erreur'}")
    st.bar_chart(
        pd.DataFrame({"probabilité": probas[ordre]}, index=[etiquettes[i] for i in ordre]),
        horizontal=True,
        color=ACC,
        height=260,
    )
