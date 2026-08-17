# Classification automatique d'articles de marketplace

Étude de faisabilité : les descriptions et les photographies déjà fournies par les vendeurs
permettent-elles de retrouver automatiquement la catégorie d'un article ?

[![ci](https://github.com/richardhugou/product-category-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/richardhugou/product-category-classifier/actions/workflows/ci.yml)

**→ [Le rapport complet](docs/RAPPORT.md)**

---

## Problématique

Sur une marketplace, chaque vendeur choisit lui-même la catégorie de son article au moment de la
mise en ligne. À mesure que le volume augmente, deux vendeurs classent des produits semblables à
deux endroits différents, et un article mal rangé devient introuvable :
l'acheteur qui filtre par catégorie ne le voit pas, et le vendeur n'apprend jamais pourquoi il ne
vend pas.

Le jeu d'étude compte 1 050 produits, 7 catégories de 150 produits, une description en anglais et une
photographie par article.

## Résultats

**Les catégories sont déjà présentes dans les données.** Sans montrer une seule étiquette, la chaîne
projette les produits en deux dimensions, un K-means forme sept groupes, et l'indice de Rand ajusté
mesure leur accord avec les vraies catégories.

| Représentation | Source | Dimensions | Accord (projection) | Accord (espace complet) |
|---|---|---|---|---|
| **CNN (VGG16)** | image | 512 | **0,510** | **0,540** |
| USE | texte | 512 | 0,440 | 0,333 |
| TF-IDF | texte | 5 000 | 0,325 | 0,214 |
| Comptage + bigrammes | texte | 5 000 | 0,316 | 0,227 |
| BERT | texte | 768 | 0,316 | 0,288 |
| Comptage de mots | texte | 2 444 | 0,306 | 0,270 |
| Word2Vec | texte | 300 | 0,300 | 0,207 |
| SIFT | image | 256 | 0,044 | 0,056 |

![Les sept projections](reports/fig5_projections.png)

Sur les **mêmes photographies**, un réseau convolutif atteint 0,510 quand SIFT reste
proche du hasard : SIFT décrit des motifs locaux, pas ce qu'est l'objet. Un **comptage simple de mots**
fait pratiquement jeu égal avec BERT : sur des fiches de spécifications, le vocabulaire est très
discriminant et la syntaxe presque absente. Et VGG16 est la seule représentation dont l'accord est
**meilleur avant réduction qu'après** : sa structure ne doit rien à t-SNE.

Un indice de 0,51 n'est pas une proportion de produits bien classés. C'est une mesure de
correspondance entre deux partitions.

**En supervisé, à partir des seules images : 137 produits sur 158 correctement classés**, soit une F1
macro de 0,867, avec un VGG16 figé et 735 images d'entraînement.

![Matrice de confusion](reports/fig8_confusion_image.png)

La data augmentation a été comparée sur le jeu de validation avant tout accès au jeu réservé. Elle
n'apporte pas d'amélioration nette : six millièmes de point, moins d'un produit sur 157. Une
augmentation forte et répétée dégrade la performance. Elle n'est pas pour autant sans effet : elle
déplace les erreurs d'une catégorie à l'autre.

**La confusion la plus tenace traverse les deux approches.** Les housses de coussin et les couettes se
regroupent avec les serviettes et les pyjamas de bébé, sans étiquettes comme avec. L'algorithme
regroupe par matière et mise en scène ; la nomenclature regroupe par usage commercial.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-encoders.txt

python faisabilite.py                  # 7 représentations, projections 2D, K-means, ARI
python supervise_image.py              # classification supervisée et data augmentation
python collecte_api.py                 # collecte « champagne » via Open Food Facts
python scripts/comparer_modernes.py    # benchmark des représentations et fusion
```

Le socle seul (`requirements.txt`) suffit pour l'exploration et les modèles classiques ;
`requirements-encoders.txt` ajoute les réseaux pré-entraînés, environ 3 Go de poids au premier
lancement. Les caractéristiques extraites sont mises en cache : une seconde exécution ne recalcule ni
SIFT ni VGG16.

Autres cibles : `make test`, `make lint`, `make notebooks`, `make help`.

## Structure

```
├── src/
│   ├── pipeline.py          chargement et découpe · source unique, appelée partout
│   ├── pretraitement.py     nettoyage du texte, harmonisation des images
│   ├── representations.py   les 7 représentations imposées par la mission
│   ├── faisabilite.py       projection ACP + t-SNE, K-means, indice de Rand ajusté
│   ├── supervise_image.py   socle VGG16 figé, data augmentation, tête de classification
│   ├── text.py              TF-IDF et encodeurs de texte
│   ├── images.py            caractéristiques visuelles, mises en cache
│   ├── fusion.py            concaténation après normalisation L2
│   ├── evaluate.py          métriques
│   └── figures.py           figures d'exploration
├── faisabilite.py           étude de faisabilité · première demande
├── supervise_image.py       classification supervisée · deuxième demande
├── collecte_api.py          collecte via API · troisième demande
├── notebooks/               les notebooks, exécutés, dans l'ordre de la mission
│   ├── 01_eda_etl · 02_visualisation · 03_representations
│   ├── 04_faisabilite · 05_supervise_image · 06_collecte_api
│   └── complementaires/     les travaux hors périmètre de la mission
├── tests/                   23 tests, dont le test anti-fuite
└── docs/RAPPORT.md          le rapport
```

**Une seule chose découpe les données** : `src/pipeline.py`, en 70 / 15 / 15 stratifié à graine fixe.
Tous les scripts l'appellent, ce qui garantit que les chiffres se comparent entre eux, et
`tests/test_pipeline.py` vérifie que les trois parts sont disjointes.

## Extensions hors périmètre

Le dépôt prolonge l'étude par un benchmark de représentations, texte et image, à découpe et
classifieur constants, puis par la fusion des deux modalités : `scripts/comparer_modernes.py`,
repris dans le rapport, parties 5 et 6. Sélection sur la validation, une seule ouverture du jeu
réservé : F1 macro 0,987, 156 produits sur 158.

Les explorations antérieures restent dans le dépôt, `benchmark.py`, `optimize.py`, l'application de
démonstration `app.py` et les carnets `notebooks/complementaires/` : protocoles différents, chiffres
non comparables à ceux du rapport.

## Organisation du dépôt

`main` porte les versions publiées, `develop` intègre, et chaque lot de travail vit sur sa propre
branche avant d'être fusionné. L'intégration continue tourne sur `main`, `develop` et toute branche
`feature/**` : lint, format, tests, puis un rejeu de la chaîne depuis un clone propre. Les réseaux
pré-entraînés en sont exclus : plusieurs gigaoctets de poids par exécution seraient disproportionnés.

## Licence

MIT.
