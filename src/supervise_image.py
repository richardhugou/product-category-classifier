"""Classification supervisée à partir des seules photographies.

L'étude de faisabilité a montré que les caractéristiques d'un réseau
pré-entraîné structurent le catalogue. On passe ici à la question suivante :
en montrant les catégories au modèle, jusqu'où va-t-on ?

Le réseau convolutif reste figé et sert d'extracteur ; seule une tête de
classification est apprise. C'est le protocole d'évaluation habituel du
transfert, et le seul raisonnable sur 735 images d'entraînement : réajuster
les 138 millions de paramètres de VGG16 sur un tel volume conduirait au
surapprentissage.

La data augmentation est appliquée aux images, avant l'extraction. Comme le
socle est figé, produire une fois pour toutes plusieurs variantes de chaque
image d'entraînement puis en extraire les caractéristiques revient au même que
de tirer une transformation différente à chaque époque, à ceci près que le
répertoire des variantes est fixe. En échange, on ne traverse le réseau qu'une
fois par variante au lieu d'une fois par époque.
"""

from __future__ import annotations

import numpy as np

from src.pipeline import SEED

TAILLE = 224
LOT = 16


def _appareil() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _socle(device: str):
    """VGG16 pré-entraîné, tête de classification retirée, poids figés."""
    import torch
    from torchvision.models import VGG16_Weights, vgg16

    reseau = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in reseau.parameters():
        p.requires_grad = False
    return (
        reseau,
        torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
    )


def transformations(intensite: str):
    """Les transformations appliquées aux images d'entraînement.

    Le choix est contraint par la nature des photographies : des produits de
    catalogue, centrés, souvent sur fond clair. Le retournement horizontal et
    de légères rotations restent plausibles : un même article peut être
    photographié dans l'autre sens. Le retournement vertical, lui, produirait
    des images qu'on ne rencontrera jamais, et n'est pas retenu.

    Deux intensités sont proposées, et la comparaison des deux fait partie du
    résultat. Conclure qu'une augmentation n'apporte rien à partir d'un seul
    réglage laisserait ouverte l'objection qu'il était mal choisi.
    """
    from torchvision import transforms

    if intensite == "aucune":
        return transforms.Compose([transforms.Resize((TAILLE, TAILLE))])
    if intensite == "douce":
        return transforms.Compose(
            [
                transforms.Resize((TAILLE, TAILLE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10, fill=255),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(TAILLE, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15, fill=255),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        ]
    )


def extraire(uniq_ids: list[str], intensite: str = "aucune", copies: int = 1):
    """Caractéristiques VGG16 des images désignées.

    Avec `copies > 1`, chaque image produit plusieurs variantes augmentées.
    Renvoie (matrice, index) : l'index dit de quelle image d'origine vient
    chaque ligne, ce qui permet de dupliquer les étiquettes en conséquence.
    """
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    from src.pretraitement import chemin_image

    device = _appareil()
    reseau, moyenne, ecart = _socle(device)
    transformer = transformations(intensite)
    torch.manual_seed(SEED)

    sorties, index = [], []
    for c in range(copies):
        for debut in range(0, len(uniq_ids), LOT):
            lot = uniq_ids[debut : debut + LOT]
            images = []
            for uid in lot:
                image = Image.open(chemin_image(uid)).convert("RGB")
                images.append(F.to_tensor(transformer(image)))
            x = torch.stack(images).to(device)
            if x.shape[-1] != TAILLE:
                x = torch.nn.functional.interpolate(x, size=(TAILLE, TAILLE), mode="bilinear")
            x = (x - moyenne) / ecart
            with torch.no_grad():
                sorties.append(reseau(x).mean(dim=(2, 3)).cpu().numpy())
            index.extend(range(debut, debut + len(lot)))
        del c
    return np.vstack(sorties).astype(np.float32), np.asarray(index)


def tete():
    """La tête de classification : 512 caractéristiques en entrée, 7 en sortie."""
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(256,),
        alpha=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=SEED,
    )
