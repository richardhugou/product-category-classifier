"""Les sept représentations demandées par la mission, derrière une interface unique.

Cinq pour le texte — comptage simple, TF-IDF, Word2Vec, BERT, USE — et deux
pour l'image — SIFT en sac de mots visuels, et un réseau convolutif
pré-entraîné utilisé en transfert. Chacune reçoit les 1 050 articles et rend
une matrice `(1050, d)` accompagnée de son temps de calcul.

L'étude de faisabilité étant non supervisée, les représentations sont ajustées
sur l'ensemble du corpus : aucune étiquette n'intervient, donc aucune fuite
n'est possible. La question posée n'est pas « sait-on prédire ? » mais « les
catégories se dessinent-elles d'elles-mêmes ? ».

Tout est mis en cache sur disque : recalculer SIFT ou VGG16 sur 1 050
photographies prend plusieurs minutes.

    python -m src.representations          # calcule et met en cache
    python -m src.representations --liste  # affiche l'état du cache
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.pipeline import SEED

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "features"
IMAGES = ROOT / "data" / "Flipkart" / "Images"

# Taille du vocabulaire visuel de SIFT. Chaque photographie produit des
# centaines de descripteurs locaux ; on les regroupe en « mots visuels » pour
# obtenir un vecteur de taille fixe, comme un sac de mots sur du texte.
MOTS_VISUELS = 256
DIM_WORD2VEC = 300


# --------------------------------------------------------------------- texte


def _sac_de_mots(textes: list[str], ngram: tuple[int, int]) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(
        lowercase=True, stop_words="english", max_features=5000, ngram_range=ngram, min_df=2
    )
    return vec.fit_transform(textes).toarray().astype(np.float32)


def comptage(textes: list[str]) -> np.ndarray:
    """Sac de mots au sens strict : combien de fois chaque mot apparaît.

    Unigrammes seulement. C'est la référence la plus simple qu'on puisse
    construire, et c'est à ce titre qu'elle sert : tout ce qui suit doit
    justifier son coût par rapport à elle.
    """
    return _sac_de_mots(textes, (1, 1))


def comptage_bigrammes(textes: list[str]) -> np.ndarray:
    """Le même comptage, mots seuls et paires de mots.

    Isole l'apport des bigrammes : seule cette option change par rapport à
    `comptage`, donc l'écart entre les deux ne peut venir que d'eux.
    """
    return _sac_de_mots(textes, (1, 2))


def tfidf(textes: list[str]) -> np.ndarray:
    """Le même comptage, pondéré par la rareté du terme dans le corpus."""
    from src.text import vectoriseur

    return vectoriseur().fit_transform(textes).toarray().astype(np.float32)


def word2vec(textes: list[str]) -> np.ndarray:
    """Vecteurs de mots appris sur le corpus, moyennés par description.

    Le modèle est entraîné ici même, sur les 1 050 descriptions. C'est peu pour
    apprendre une sémantique, et le résultat le dira.
    """
    from gensim.models import Word2Vec

    from src.pretraitement import jetons

    corpus = [jetons(t) for t in textes]
    modele = Word2Vec(
        corpus,
        vector_size=DIM_WORD2VEC,
        window=5,
        min_count=2,
        workers=1,
        seed=SEED,
        epochs=30,
    )
    connus = modele.wv
    sorties = []
    for doc in corpus:
        vecteurs = [connus[m] for m in doc if m in connus]
        sorties.append(np.mean(vecteurs, axis=0) if vecteurs else np.zeros(DIM_WORD2VEC))
    return np.vstack(sorties).astype(np.float32)


def bert(textes: list[str]) -> np.ndarray:
    """Représentation contextuelle figée, moyenne des jetons."""
    from src.text import charger_encodeur, encoder

    tok, mdl, device = charger_encodeur("bert-base-uncased")
    return encoder(textes, tok, mdl, device).astype(np.float32)


URL_USE = "https://tfhub.dev/google/universal-sentence-encoder/4"

# Encodage USE, exécuté dans un interpréteur qui n'a jamais importé PyTorch.
_SCRIPT_USE = """
import json, sys
import numpy as np
import tensorflow_hub as hub

textes = json.load(open(sys.argv[1], encoding="utf-8"))
modele = hub.load({url!r})
vecteurs = np.vstack([modele(textes[i:i + 32]).numpy() for i in range(0, len(textes), 32)])
np.save(sys.argv[2], vecteurs.astype(np.float32))
"""


def use(textes: list[str]) -> np.ndarray:
    """Universal Sentence Encoder — l'implémentation de référence.

    C'est bien le modèle publié par Google sur TensorFlow Hub, et non une
    distillation approchante : la mission le nomme explicitement.

    L'encodage est délégué à un sous-processus, pour une raison très concrète.
    Chargé dans un interpréteur ayant déjà importé PyTorch — ce que font BERT,
    SIFT et VGG16 — TensorFlow se bloque sur cette machine : le processus reste
    figé indéfiniment, sans lever la moindre erreur. Isolé, il charge le modèle
    en deux secondes. Le sous-processus n'importe donc que TensorFlow, écrit le
    résultat sur disque, et rend la main.
    """
    import json
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as dossier:
        entree = Path(dossier) / "textes.json"
        sortie = Path(dossier) / "use.npy"
        entree.write_text(json.dumps(textes), encoding="utf-8")
        subprocess.run(
            [sys.executable, "-c", _SCRIPT_USE.format(url=URL_USE), str(entree), str(sortie)],
            check=True,
        )
        return np.load(sortie)


# --------------------------------------------------------------------- image


def sift(uniq_ids: list[str]) -> np.ndarray:
    """Points d'intérêt SIFT regroupés en sac de mots visuels.

    SIFT décrit chaque point remarquable d'une image par 128 nombres, mais leur
    nombre varie d'une photographie à l'autre. Pour obtenir un vecteur de
    taille fixe, on regroupe tous les descripteurs du corpus en un vocabulaire
    visuel, puis on compte combien de points de chaque image tombent dans
    chaque mot. C'est exactement la logique du sac de mots sur du texte.
    """
    import cv2
    from sklearn.cluster import MiniBatchKMeans

    from src.pretraitement import charger_image

    detecteur = cv2.SIFT_create()
    par_image: list[np.ndarray | None] = []
    for uid in uniq_ids:
        gris = charger_image(uid, gris=True)
        _, descripteurs = detecteur.detectAndCompute(gris, None)
        par_image.append(descripteurs)

    trouves = [d for d in par_image if d is not None]
    print(f"    {sum(len(d) for d in trouves)} descripteurs sur {len(trouves)} images")

    # Le vocabulaire visuel est appris sur un échantillon : tout garder ferait
    # plusieurs centaines de milliers de vecteurs pour un gain nul.
    tous = np.vstack(trouves)
    rng = np.random.default_rng(SEED)
    echantillon = tous[rng.choice(len(tous), size=min(100_000, len(tous)), replace=False)]
    kmeans = MiniBatchKMeans(
        n_clusters=MOTS_VISUELS, random_state=SEED, n_init=3, batch_size=2048
    ).fit(echantillon)

    histogrammes = np.zeros((len(uniq_ids), MOTS_VISUELS), dtype=np.float32)
    for i, d in enumerate(par_image):
        if d is None:
            continue
        for mot in kmeans.predict(d):
            histogrammes[i, mot] += 1
    # Normaliser : une photographie riche en détails ne doit pas dominer.
    sommes = histogrammes.sum(axis=1, keepdims=True)
    return histogrammes / np.maximum(sommes, 1)


def cnn(uniq_ids: list[str]) -> np.ndarray:
    """VGG16 pré-entraîné sur ImageNet, tête de classification retirée.

    On garde la sortie du dernier étage convolutif après agrégation, soit 512
    nombres par image. Le réseau n'est pas réentraîné : il sert d'extracteur.
    """
    import torch
    from torchvision.models import VGG16_Weights, vgg16

    from src.pretraitement import charger_image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    reseau = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    moyenne = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    ecart = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    sorties = []
    with torch.no_grad():
        for i in range(0, len(uniq_ids), 16):
            lot = np.stack([charger_image(u) for u in uniq_ids[i : i + 16]])
            x = torch.from_numpy(lot).permute(0, 3, 1, 2).float().to(device) / 255.0
            x = (x - moyenne) / ecart
            cartes = reseau(x)  # (b, 512, 7, 7)
            sorties.append(cartes.mean(dim=(2, 3)).cpu().numpy())
    return np.vstack(sorties).astype(np.float32)


# ------------------------------------------------------------------ registre

TEXTE = {
    "Comptage de mots": comptage,
    "Comptage + bigrammes": comptage_bigrammes,
    "TF-IDF": tfidf,
    "Word2Vec": word2vec,
    "BERT": bert,
    "USE": use,
}

IMAGE = {
    "SIFT": sift,
    "CNN (VGG16)": cnn,
}


def _fichier(nom: str) -> Path:
    return CACHE / f"repr_{nom.lower().replace(' ', '_').replace('(', '').replace(')', '')}.npy"


def obtenir(nom: str, entrees) -> tuple[np.ndarray, float]:
    """Renvoie (matrice, secondes). Le temps vaut 0 si le cache a servi."""
    CACHE.mkdir(parents=True, exist_ok=True)
    chemin = _fichier(nom)
    if chemin.exists():
        return np.load(chemin), 0.0

    fonction = {**TEXTE, **IMAGE}[nom]
    t0 = time.perf_counter()
    X = fonction(list(entrees))
    secondes = time.perf_counter() - t0
    np.save(chemin, X)
    return X, secondes


def toutes(textes: list[str], uniq_ids: list[str]) -> dict[str, np.ndarray]:
    """Les sept représentations, calculées ou relues, dans l'ordre du rapport."""
    resultat = {}
    for nom in TEXTE:
        X, s = obtenir(nom, textes)
        print(f"  {nom:20s} {X.shape[1]:6d} dimensions" + (f" · {s:.0f} s" if s else " · cache"))
        resultat[nom] = X
    for nom in IMAGE:
        X, s = obtenir(nom, uniq_ids)
        print(f"  {nom:20s} {X.shape[1]:6d} dimensions" + (f" · {s:.0f} s" if s else " · cache"))
        resultat[nom] = X
    return resultat


if __name__ == "__main__":
    import sys

    from src.pipeline import TEXT_COL, load

    df = load()
    if "--liste" in sys.argv:
        for nom in {**TEXTE, **IMAGE}:
            f = _fichier(nom)
            print(f"  {nom:20s} {'en cache' if f.exists() else 'à calculer'}  {f.name}")
    else:
        toutes(df[TEXT_COL].tolist(), df["uniq_id"].tolist())
