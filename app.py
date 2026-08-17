"""Démonstration : catégoriser un article par le texte, par l'image, ou par les deux.

    streamlit run app.py

Les trois modèles tournent côte à côte sur le même article : c'est le seul
moyen de voir *où* l'image apporte quelque chose. Le seuil de confiance
matérialise UC4 : sous le seuil, le modèle ne tranche pas et l'article part en
revue humaine.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.images import image_path  # noqa: E402
from src.pipeline import LABEL_COL, TEXT_COL, load, split  # noqa: E402

MODELS = ROOT / "models"
FEATURES = ROOT / "data" / "features"
SEUIL_DEFAUT = 0.60
MS_ENCODAGE_IMAGE = 35.7  # mesuré sur la machine de développement

# Trois premiers créneaux d'une palette catégorielle validée : ce sont les
# seuls à rester distinguables en vision des couleurs déficiente lorsque
# toutes les paires se côtoient, ce qui est le cas d'un graphe groupé.
COULEURS = {
    "Texte seul": "#2a78d6",
    "Image seule": "#eb6834",
    "Texte + image": "#1baf7a",
}
# Encres neutres : elles portent le texte et les repères, jamais l'identité
# d'une série. Toutes deux passent le contraste sur fond clair comme sur sombre.
ENCRE_FORTE = "#f2f4f7"
ENCRE_DOUCE = "#98a2b3"

st.set_page_config(page_title="Catégorisation d'articles", page_icon="🏷️", layout="wide")


def _empreinte(p: Path) -> str:
    """Version du modèle : REQ-08. Toute prédiction est rattachable à son modèle."""
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


@st.cache_resource
def get_modeles():
    """Renvoie {nom: (modèle, version)} pour ceux qui existent sur disque."""
    m = {}
    vec, texte = joblib.load(MODELS / "tfidf_mlp.joblib")
    m["Texte seul"] = (texte, _empreinte(MODELS / "tfidf_mlp.joblib"))
    if (MODELS / "image_mlp.joblib").exists():
        m["Image seule"] = (
            joblib.load(MODELS / "image_mlp.joblib"),
            _empreinte(MODELS / "image_mlp.joblib"),
        )
    if (MODELS / "fusion_mlp.joblib").exists():
        _, fusion = joblib.load(MODELS / "fusion_mlp.joblib")
        m["Texte + image"] = (fusion, _empreinte(MODELS / "fusion_mlp.joblib"))
    return vec, m


@st.cache_resource
def get_encodeur_image():
    """Chargé seulement si une photographie est téléversée : 346 Mo."""
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
        return cache[uniq_id], MS_ENCODAGE_IMAGE  # le coût réel est réintégré
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


vec, modeles = get_modeles()
test = get_exemples()
labels = sorted(test[LABEL_COL].unique())

st.title("Catégorisation automatique d'articles")
st.caption(
    "7 catégories · entraîné sur 735 articles, évalué sur 158 jamais vus · "
    "seuil métier F1 macro ≥ 0,90"
)

# ─────────────────────────────────────────────────────────── barre latérale
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
st.sidebar.caption("**Modèles chargés**")
for nom, (_, version) in modeles.items():
    st.sidebar.caption(f"{nom} · `{version}`")

# ───────────────────────────────────────────────────────────────── saisie
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


def graphe_probabilites(series: dict[str, np.ndarray], vrai: str | None):
    """Barres groupées horizontales : une barre par modèle et par catégorie.

    Surtout pas d'empilement : additionner les probabilités de deux modèles
    n'a aucun sens, et la hauteur cumulée se lirait comme une valeur.
    """
    lignes = [
        {"Catégorie": c, "Modèle": nom, "Probabilité": float(p)}
        for nom, proba in series.items()
        for c, p in zip(labels, proba, strict=True)
    ]
    d = pd.DataFrame(lignes)

    reference = series.get("Texte + image", next(iter(series.values())))
    ordre = [labels[i] for i in np.argsort(-reference)]
    noms = list(series.keys())

    # Le repère de vérité terrain vit sur l'étiquette d'axe et cumule trois
    # signaux : un chevron, la graisse et un contraste maximal. Aucun n'est une
    # teinte : une couleur d'annotation ne doit jamais pouvoir se lire comme une
    # série, et l'or que j'avais d'abord retenu se confondait avec l'orange de
    # « Image seule », a fortiori sous déficience de perception rouge-vert.
    if vrai:
        echappe = vrai.replace("'", "\\'")
        est_vrai = f"datum.value === '{echappe}'"
        axe_y = alt.Axis(
            labelFontSize=12,
            labelLimit=230,
            labelExpr=f"{est_vrai} ? '▶  ' + datum.value : datum.value",
            labelFontWeight=alt.expr(f"{est_vrai} ? 'bold' : 'normal'"),
            labelColor=alt.expr(f"{est_vrai} ? '{ENCRE_FORTE}' : '{ENCRE_DOUCE}'"),
        )
    else:
        axe_y = alt.Axis(labelFontSize=12, labelLimit=230, labelColor=ENCRE_DOUCE)

    base = alt.Chart(d).encode(
        # paddingInner sépare les trois barres d'un même groupe : sans lui les
        # aplats se touchent et le groupe se lit comme un seul bloc.
        y=alt.Y("Catégorie:N", sort=ordre, title=None, axis=axe_y),
        yOffset=alt.YOffset("Modèle:N", sort=noms, scale=alt.Scale(paddingInner=0.25)),
    )
    barres = base.mark_bar(cornerRadiusEnd=3).encode(
        # Cinq graduations suffisent à situer une proportion. La version
        # précédente en affichait vingt-et-une, avec une grille à chaque cran :
        # l'œil comptait des lignes au lieu de comparer des longueurs.
        x=alt.X(
            "Probabilité:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(
                format=".0%",
                title=None,
                values=[0, 0.25, 0.5, 0.75, 1],
                grid=True,
                gridOpacity=0.18,
                domain=False,
                tickSize=0,
                labelColor=ENCRE_DOUCE,
            ),
        ),
        color=alt.Color(
            "Modèle:N",
            sort=noms,
            scale=alt.Scale(domain=noms, range=[COULEURS[n] for n in noms]),
            legend=alt.Legend(
                orient="top", title=None, symbolType="square", labelColor=ENCRE_DOUCE
            ),
        ),
        tooltip=[
            alt.Tooltip("Catégorie:N"),
            alt.Tooltip("Modèle:N"),
            alt.Tooltip("Probabilité:Q", format=".1%"),
        ],
    )
    # Étiquettes directes : avec trois séries, l'œil ne compare pas des
    # longueurs proches de façon fiable. On étiquette les trois barres de chaque
    # groupe ou aucune : un étiquetage au cas par cas se lit comme un oubli.
    etiquettes = base.mark_text(align="left", dx=5, fontSize=11, color=ENCRE_DOUCE).encode(
        x=alt.X("Probabilité:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("Probabilité:Q", format=".0%"),
        opacity=alt.condition(alt.datum.Probabilité >= 0.03, alt.value(1), alt.value(0)),
    )

    # Pas de couche de fond pour marquer la vérité terrain : un mark_rect
    # superposé perturbe le calcul des bandes quand un yOffset est en jeu, et
    # les barres finissent par se chevaucher.
    return (barres + etiquettes).properties(height=alt.Step(19))


# ─────────────────────────────────────────────────────────────── prédiction
if st.button("Catégoriser", type="primary") and texte.strip():
    X = vec.transform([texte])
    v_img, ms_img = vecteur_image(uniq_id=None if televerse else uniq, fichier=televerse)

    resultats: dict[str, tuple] = {}  # nom -> (proba, ms, version)

    modele, version = modeles["Texte seul"]
    t0 = time.perf_counter()
    p = modele.predict_proba(X)[0]
    resultats["Texte seul"] = (p, (time.perf_counter() - t0) * 1000, version)

    if v_img is not None:
        from scipy.sparse import csr_matrix, hstack
        from sklearn.preprocessing import normalize

        img_norme = normalize(v_img.reshape(1, -1))

        if "Image seule" in modeles:
            modele, version = modeles["Image seule"]
            t0 = time.perf_counter()
            p = modele.predict_proba(img_norme)[0]
            resultats["Image seule"] = (
                p,
                (time.perf_counter() - t0) * 1000 + ms_img,
                version,
            )

        if "Texte + image" in modeles:
            modele, version = modeles["Texte + image"]
            t0 = time.perf_counter()
            p = modele.predict_proba(hstack([X, csr_matrix(img_norme)]).tocsr())[0]
            resultats["Texte + image"] = (
                p,
                (time.perf_counter() - t0) * 1000 + ms_img,
                version,
            )

    vrai = st.session_state.get("vrai") if st.session_state.get("texte") == texte else None

    st.markdown("### Ce que chaque modèle propose")
    colonnes = st.columns(len(resultats))
    verdicts = {}
    for col, (nom, (proba, ms, version)) in zip(colonnes, resultats.items(), strict=True):
        i = int(proba.argmax())
        categorie, confiance = labels[i], float(proba[i])
        verdicts[nom] = categorie
        with col:
            # Bloc composé à la main plutôt que st.metric + st.progress :
            #   : st.metric tronque les valeurs longues, et « Home Decor &
            #     Festive Needs » est précisément la bonne réponse ;
            #   : st.progress ne se colore pas, si bien que la jauge restait
            #     bleue sous une pastille orange : la couleur cessait de
            #     désigner la série.
            teinte = COULEURS[nom]
            retenu = confiance >= seuil
            st.markdown(
                f"""
<div style="line-height:1.35">
  <div style="font-weight:700;font-size:.95rem">
    <span style="display:inline-block;width:10px;height:10px;border-radius:2px;
                 background:{teinte};margin-right:7px"></span>{nom}
  </div>
  <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
              color:{ENCRE_DOUCE};margin-top:.7rem">Catégorie proposée</div>
  <div style="font-size:1.35rem;font-weight:700;margin:.15rem 0 .55rem">
    {categorie if retenu else "En revue"}</div>
  <div style="height:7px;border-radius:4px;background:{ENCRE_DOUCE}33;overflow:hidden">
    <div style="width:{confiance * 100:.1f}%;height:100%;background:{teinte}"></div>
  </div>
  <div style="font-size:.78rem;color:{ENCRE_DOUCE};margin-top:.3rem">
    confiance {confiance:.0%}{"" if retenu else f" · sous le seuil de {seuil:.0%}"}</div>
</div>""",
                unsafe_allow_html=True,
            )
            if not retenu:
                st.caption(f"Sans le garde-fou, la proposition serait « {categorie} ».")
            st.caption(f"{ms:.1f} ms · modèle `{version}`")
            if vrai:
                st.caption("Correct ✓" if categorie == vrai else "Erroné ✗")

    if vrai:
        justes = [n for n, c in verdicts.items() if c == vrai]
        st.info(
            f"**Catégorie réelle : {vrai}**, "
            + (f"trouvée par : {', '.join(justes)}." if justes else "aucun modèle ne la trouve.")
        )

    st.markdown("### Probabilité attribuée à chaque catégorie")
    st.caption(
        "Une barre par modèle et par catégorie. Les barres ne sont pas empilées : "
        "additionner les probabilités de deux modèles n'aurait aucun sens."
        + (
            " La catégorie marquée d'un chevron sur l'axe est la vérité terrain : "
            "la référence contre laquelle les trois modèles se jugent."
            if vrai
            else ""
        )
    )
    st.altair_chart(
        graphe_probabilites({n: r[0] for n, r in resultats.items()}, vrai),
        use_container_width=True,
    )

    # Une vue tabulaire double le graphe : les valeurs exactes ne doivent pas
    # dépendre du survol, sans quoi elles sont perdues au clavier et à
    # l'impression.
    with st.expander("Les mêmes valeurs en tableau"):
        table = pd.DataFrame({nom: proba for nom, (proba, _, _) in resultats.items()}, index=labels)
        table = table.loc[table.iloc[:, -1].sort_values(ascending=False).index]
        if vrai:
            table.index = [f"▶ {c}" if c == vrai else c for c in table.index]
        st.dataframe(
            table.style.format("{:.1%}"),
            use_container_width=True,
        )
        if vrai:
            st.caption("Le chevron marque la vérité terrain.")

# ─────────────────────────────────────────────────────────────── benchmark
with st.expander("Le benchmark complet"):
    p = ROOT / "reports" / "benchmark.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)
        st.caption(
            "Cinq modèles sur six franchissent le seuil métier : il ne départage donc rien. "
            "Le critère de décision devient le coût. La fusion gagne 3 points de F1, et "
            "surtout 8 points sur la catégorie la plus faible : pour 600 fois le coût "
            "d'inférence du modèle texte."
        )
    else:
        st.info("Lancer `python benchmark.py --encoders --images` pour produire le tableau.")
