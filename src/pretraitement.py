"""Prétraitement du texte et des images.

Les étapes sont isolées plutôt que fondues dans une seule fonction, pour deux
raisons. La première est qu'on peut les vérifier une par une. La seconde est
qu'on peut les montrer : `etapes_texte` rend l'état intermédiaire après chaque
opération, ce dont le rapport se sert pour suivre un article réel du texte brut
jusqu'aux jetons.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "Flipkart" / "Images"

TAILLE_CNN = 224
TAILLE_SIFT = 256

# Le corpus contient au moins une photographie de 93 mégapixels. Pillow refuse
# par défaut d'ouvrir les images de cette taille, par précaution contre les
# fichiers malveillants. La source est connue et maîtrisée : on lève la garde.
Image.MAX_IMAGE_PIXELS = None

MOTS_OUTILS = None  # chargé paresseusement depuis scikit-learn


def _mots_outils() -> frozenset[str]:
    global MOTS_OUTILS
    if MOTS_OUTILS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        MOTS_OUTILS = ENGLISH_STOP_WORDS
    return MOTS_OUTILS


# ---------------------------------------------------------------------- texte


def etapes_texte(texte: str) -> dict[str, object]:
    """Les états successifs d'une description. Sert à illustrer la chaîne."""
    minuscules = str(texte).lower()
    sans_ponctuation = re.sub(r"[^a-z0-9\s]", " ", minuscules)
    mots = sans_ponctuation.split()
    retenus = [m for m in mots if m not in _mots_outils() and len(m) > 2]
    return {
        "brut": str(texte),
        "minuscules": minuscules,
        "sans ponctuation": re.sub(r"\s+", " ", sans_ponctuation).strip(),
        "mots": mots,
        "jetons": retenus,
    }


def jetons(texte: str) -> list[str]:
    """La sortie de la chaîne : la liste de mots retenus."""
    return etapes_texte(texte)["jetons"]  # type: ignore[return-value]


# ---------------------------------------------------------------------- image


def chemin_image(uniq_id: str) -> Path:
    return IMAGES / f"{uniq_id}.jpg"


def charger_image(uniq_id: str, taille: int | None = None, gris: bool = False) -> np.ndarray:
    """Ouvre une photographie et l'harmonise.

    Les fichiers du corpus n'ont ni la même taille ni le même rapport de forme.
    Chaque méthode impose les siens : le réseau convolutif attend des carrés de
    224 pixels normalisés comme ceux de son entraînement, SIFT travaille sur
    des niveaux de gris et gagne à ce que le contraste soit égalisé.
    """
    image = Image.open(chemin_image(uniq_id))
    if gris:
        image = ImageOps.equalize(ImageOps.grayscale(image))
        image = image.resize((taille or TAILLE_SIFT,) * 2, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)
    image = image.convert("RGB").resize((taille or TAILLE_CNN,) * 2, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def dimensions_brutes(uniq_ids) -> list[tuple[int, int]]:
    """Tailles d'origine, pour documenter l'hétérogénéité du corpus."""
    return [Image.open(chemin_image(u)).size for u in uniq_ids]


if __name__ == "__main__":
    from src.pipeline import TEXT_COL, load

    df = load()
    ligne = df.iloc[3]
    etapes = etapes_texte(ligne[TEXT_COL])
    print("brut          :", etapes["brut"][:100], "…")
    print("minuscules    :", etapes["minuscules"][:100], "…")
    print("mots          :", len(etapes["mots"]), "mots")
    print("jetons        :", len(etapes["jetons"]), "retenus")
    print("              :", etapes["jetons"][:15])
    img = charger_image(ligne["uniq_id"])
    print(f"image         : {img.shape}, {img.dtype}")
