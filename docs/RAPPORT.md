# Étude de faisabilité d'un moteur de classification automatique d'articles

**À partir des descriptions textuelles et des photographies de produits**

Richard Hugou · août 2026

---

# 1. Mission et objectif

L'entreprise « Place de marché » prépare une marketplace anglophone où chaque vendeur publie une
photographie, une description, et choisit lui-même la catégorie de son article. Ce fonctionnement ne
tient pas à l'échelle : classements divergents, produits invisibles au filtrage. L'automatisation
doit précéder la croissance du catalogue.

Question posée par Linda, Lead Data Scientist : les informations déjà présentes dans une fiche
produit permettent-elles de retrouver automatiquement sa catégorie ?

Trois demandes structurent l'étude :

1. **Faisabilité** : représenter texte et image par des méthodes imposées (comptage, TF-IDF,
   Word2Vec, BERT, USE ; SIFT, réseau convolutif en transfert), projeter, mesurer l'accord entre
   clusters et catégories.
2. **Classification supervisée** à partir des images, optimisée par data augmentation.
3. **Collecte** de produits d'épicerie fine via une API publique.

Périmètre : un échantillon de 1 050 produits, des moyens de calcul locaux. Les conclusions valent
pour ce cadre.

---

# 2. Données

**Exemple de référence : V9 METAL STRAP Analog Watch.** Conservé tout au long de l'étude.

| Champ | Contenu |
|---|---|
| `product_name` | V9 METAL STRAP Analog Watch – For Men |
| `description` | *Specifications of V9 METAL STRAP Analog Watch – For Men. General Type Analog. Style Code METAL STRAP. Occasion Casual. Ideal For Men. Warranty NO. Body Features Dial Shape Round. Strap Color STEEL. Dial Color BLACK.* |
| `product_category_tree` | `["Watches >> Wrist Watches >> V9 Wrist Watches >> ..."]` |
| `image` | photographie de 1 152 × 1 816 pixels |

Corpus : 1 050 articles, 7 catégories de 150 exactement (*Baby Care*, *Beauty and Personal Care*,
*Computers*, *Home Decor & Festive Needs*, *Home Furnishing*, *Kitchen & Dining*, *Watches*). La
cible est le premier niveau de l'arborescence `product_category_tree`. Elle est déclarée par les
vendeurs : la référence d'évaluation est donc elle-même faillible (repris en partie 8).

Constats sur les entrées :

- Descriptions : des formulaires de spécifications aplatis, non des textes rédigés. 13 à 587 mots,
  médiane 44. Longueur médiane variable selon la catégorie : *Home Furnishing* 24 mots ;
  *Kitchen & Dining* 88.
- Photographies : 890 tailles distinctes sur 1 050 fichiers ; ratios de 0,23 à 4,36 ; maximum
  93 mégapixels. Harmonisation selon la méthode consommatrice.

![Équilibre des classes et distribution des longueurs de description](../reports/fig4_donnees.png)

**Champ `brand` exclu** : 32 % de valeurs manquantes, absence fortement corrélée à la catégorie.
95 % des 338 absences se concentrent sur trois catégories.

| Catégorie | Marque absente |
|---|---|
| Watches | 140 / 150 |
| Beauty and Personal Care | 109 / 150 |
| Kitchen & Dining | 71 / 150 |
| Baby Care | 16 / 150 |
| Home Decor & Festive Needs | 2 / 150 |
| Computers | 0 / 150 |
| Home Furnishing | 0 / 150 |

Un modèle recevant ce champ apprendrait une habitude de saisie, pas le produit. Exclusion,
indicateur d'absence compris.

**Prétraitement.** Texte : minuscules · suppression ponctuation et chiffres isolés · découpage ·
retrait des mots-outils anglais. Sur l'exemple de référence : 34 mots bruts → 29 tokens. Ni
racinisation ni lemmatisation : la troncature détruirait des références de modèles (`7007YL08`),
souvent les termes les plus discriminants. Image : RGB · 224 × 224 · normalisation ImageNet pour le
réseau convolutif ; niveaux de gris · égalisation · 256 × 256 pour SIFT.

---

# 3. Faisabilité non supervisée

Protocole : les catégories ne servent qu'à colorier les graphiques et à mesurer l'accord final,
jamais à construire les représentations, ajustées sur les 1 050 produits. Étape descriptive de
faisabilité sur tout le corpus ; l'évaluation indépendante d'un modèle supervisé vient ensuite,
avec son propre protocole.

| Représentation | Modalité | Dimensions | Principe |
|---|---|---|---|
| Comptage de mots | texte | 2 444 | occurrences par terme · référence rudimentaire |
| Comptage + bigrammes | texte | 5 000 | ajout des paires de mots |
| TF-IDF | texte | 5 000 | comptage pondéré par la rareté |
| Word2Vec | texte | 300 | moyenne de vecteurs de mots, appris sur le corpus |
| BERT | texte | 768 | encodeur contextuel pré-entraîné, figé, moyenné |
| USE | texte | 512 | encodeur de phrase pré-entraîné (annexe C) |
| SIFT + BoVW | image | 256 | points d'intérêt locaux → 256 mots visuels |
| CNN (VGG16) | image | 512 | réseau pré-entraîné ImageNet, tête retirée |

Effet de la pondération TF-IDF sur l'exemple de référence : `color` (présent partout) sort du
classement ; `metal` (0,272) et `strap` (0,246) dominent.

![La chaîne de transformation, sur un article réel](../reports/fig3_transformations.png)

**Projection.** ACP à 50 composantes puis t-SNE. Chaque produit est préalablement ramené à une
longueur unitaire : la standardisation par dimension, testée d'abord, amplifiait les termes rares
au point de ramener l'accord TF-IDF à 0,001 (partie 9). Lecture t-SNE : proximités significatives,
échelle des axes non.

![Les sept projections, couleur : catégorie réelle](../reports/fig5_projections.png)

Structure partielle pour les représentations textuelles ; projection VGG16 la plus organisée ;
nuage homogène pour SIFT.

**Mesure.** Catégories masquées · K-means à 7 groupes · indice de Rand ajusté (1 : partitions
identiques ; 0 : accord équivalent au hasard). Règle de lecture : **un indice de 0,51 ne signifie
pas que 51 % des produits sont correctement catégorisés**, c'est une correspondance entre deux
partitions, pas une proportion. Mesure rapportée deux fois : sur la projection et sur la
représentation complète, t-SNE déformant.

| Représentation | Source | Dimensions | Accord sur la projection | Accord avant réduction |
|---|---|---|---|---|
| **CNN (VGG16)** | image | 512 | **0,510** | **0,540** |
| USE | texte | 512 | 0,440 | 0,333 |
| TF-IDF | texte | 5 000 | 0,325 | 0,214 |
| Comptage + bigrammes | texte | 5 000 | 0,316 | 0,227 |
| BERT | texte | 768 | 0,316 | 0,288 |
| Comptage de mots | texte | 2 444 | 0,306 | 0,270 |
| Word2Vec | texte | 300 | 0,300 | 0,207 |
| SIFT | image | 256 | 0,044 | 0,056 |

![Accord entre les groupes trouvés et les catégories réelles](../reports/fig7_ari.png)

Constats :

- Sur les mêmes photographies, VGG16 atteint 0,510 ; SIFT reste proche du hasard. SIFT décrit des
  motifs locaux, VGG16 des objets : une méthode reconnue peut répondre à une autre question que
  celle posée.
- VGG16 est la seule représentation meilleure avant réduction qu'après : la séparation existe dans
  l'espace d'origine.
- Côté texte, USE se détache (0,440) ; le comptage simple fait jeu égal avec BERT : vocabulaire
  discriminant, syntaxe quasi absente. Word2Vec, appris sur 1 050 descriptions, arrive dernier.
  Bigrammes : +0,010 en projection, −0,043 en espace complet.

![VGG16 : catégories réelles à gauche, groupes formés sans étiquettes à droite](../reports/fig6_clusters.png)

Correspondance groupe-catégorie de un à un pour VGG16 (informatique 87 %, montres 86 %, beauté
80 %). Deux zones de confusion : un groupe de textiles imprimés photographiés à plat mêle *Home
Furnishing* et *Baby Care* (69 produits déplacés) ; 31 objets décoratifs de *Home Decor* partent
vers l'ameublement. Regroupement par matière et mise en scène d'un côté, par usage commercial de
l'autre.

Conclusion de l'étape : l'information nécessaire à la catégorisation est présente dans les données
fournies. Dans cette étude non supervisée, l'image traitée par un réseau pré-entraîné en est la
source la plus prometteuse.

---

# 4. Classification supervisée des images

Protocole : découpe stratifiée figée des 1 050 produits en 735 entraînement · 157 validation · 158
test. Toute décision est prise sur la validation ; le jeu de test n'est ouvert qu'une fois, pour le
seul modèle retenu. Cette règle corrige une erreur commise en cours d'étude : une première
comparaison des stratégies avait été lue sur le test (partie 9).

Modèle : VGG16 figé en extracteur (512 caractéristiques) + tête de classification apprise. Choix
méthodologique : au vu du volume (735 images) et de l'objectif de comparaison, l'extracteur figé
limite le surapprentissage et isole la qualité de la représentation ; le réglage fin relèverait
d'une expérimentation dédiée. Sur l'exemple de référence (jeu de test, avant augmentation) :
*Watches*, probabilité 0,977.

**Data augmentation.** Quatre configurations, sur transformations plausibles pour des photographies
de catalogue :

| Stratégie | Images d'entraînement | F1 macro (validation) |
|---|---|---|
| Augmentation douce ×4 | 3 675 | **0,828** |
| Augmentation forte ×4 | 3 675 | 0,827 |
| Sans augmentation | 735 | 0,822 |
| Augmentation forte ×8 | 6 615 | 0,815 |

Écart maximal : 0,006 point de F1 macro. Insuffisant, sur cet échantillon de validation, pour
conclure à une amélioration nette. La lecture par catégorie est plus informative : l'augmentation
déplace les erreurs.

| Catégorie | Sans | Douce ×4 | Forte ×4 | Forte ×8 |
|---|---|---|---|---|
| Baby Care | 0,750 | 0,762 | 0,762 | **0,818** |
| Home Decor & Festive Needs | 0,739 | 0,776 | **0,792** | 0,773 |
| Watches | 0,818 | 0,851 | **0,864** | 0,818 |
| Beauty and Personal Care | 0,810 | 0,800 | 0,810 | **0,829** |
| Kitchen & Dining | 0,920 | 0,913 | **0,939** | 0,898 |
| Computers | **0,810** | 0,809 | 0,783 | 0,750 |
| Home Furnishing | **0,905** | 0,884 | 0,837 | 0,818 |

![L'augmentation déplace les erreurs plutôt qu'elle ne les supprime](../reports/fig9_augmentation_par_classe.png)

Gains et pertes se compensent : une augmentation uniforme n'est probablement pas la bonne
stratégie. Configuration retenue selon la règle fixée d'avance : augmentation douce ×4, avantage
non établi.

**Évaluation finale.** Jeu de test, une seule ouverture : **137 / 158 produits correctement
classés · exactitude 86,7 % · F1 macro 0,867**. Extracteur figé, sans texte.

![Matrice de confusion sur le jeu réservé](../reports/fig8_confusion_image.png)

La confusion *Baby Care* ↔ *Home Furnishing* (2 + 2 erreurs) reproduit celle de l'étude non
supervisée : ambiguïté réelle entre certaines images, réduite par la supervision, non éliminée.

---

# 5. Extension expérimentale : benchmark de représentations

Au-delà des méthodes demandées, deux questions sont mesurées sur le même protocole : des
extracteurs pré-entraînés plus récents produisent-ils de meilleures représentations sur ce
corpus ? Et que vaut la combinaison des deux modalités ?

Protocole : découpe identique 735 / 157 / 158 · extracteurs figés · une architecture de classifieur
unique, MLP à une couche cachée de 256, entraîné indépendamment derrière chaque représentation.
Un écart entre deux lignes ne peut provenir que de la représentation. Sélection au F1 macro sur la
validation ; le test n'intervient jamais dans la sélection.

| Représentation | Modalité | Dimensions | F1 macro (validation) |
|---|---|---|---|
| TF-IDF | texte | 4 532 | 0,9365 |
| DINOv2 figé (Vision Transformer) | image | 1 536 | 0,9118 |
| BERT figé | texte | 768 | 0,9050 |
| ModernBERT figé | texte | 768 | 0,9035 |
| VGG16 figé (CNN) | image | 512 | 0,8216 |

Constats :

- Texte : ModernBERT ≈ BERT (écart 0,0015) dans ce régime figé ; TF-IDF les devance tous deux,
  meilleure performance texte, cohérente avec le vocabulaire discriminant relevé en partie 3.
- Image : DINOv2 dépasse VGG16 de neuf points, mêmes photographies, mêmes conditions.
- Le VGG16 du benchmark (0,8216) reproduit exactement le « sans augmentation » de la partie 4 :
  les deux volets se recoupent.

Analyses complémentaires (pooling, longueur de contexte, similarité entre fiches) :
`scripts/ablation_modernes.py`, mesures sur validation uniquement.

![F1 par catégorie, sur la validation, pour les six configurations](../reports/fig10_comparaison_par_classe.png)

---

# 6. Fusion multimodale et évaluation finale

Principe : concaténation des deux meilleures représentations par modalité, TF-IDF (4 532) et
DINOv2 normalisé ligne à ligne (1 536), soit 6 068 caractéristiques, même architecture MLP 256.

Validation : **fusion 0,9366 · TF-IDF seul 0,9365**. Quasi-égalité : aucun gain de la
multimodalité établi sur ce corpus. La règle de sélection fixée d'avance (meilleur F1 macro de
validation) désigne la fusion.

Évaluation finale, jeu de test, une seule ouverture : **F1 macro 0,987 · 156 / 158 produits
correctement classés**.

Les deux erreurs : un lit king size étiqueté *Beauty and Personal Care*, lu *Home Furnishing*,
étiquette probablement incohérente, limite de l'étalon déclaré par les vendeurs ; un sticker mural
lu *Baby Care*, l'ambiguïté d'univers domestique observée depuis la partie 3.

---

# 7. Collecte de nouveaux produits via une API

Objectif : éprouver la collecte d'épicerie fine en vue d'un élargissement de gamme. Source
retenue : Open Food Facts, sans inscription ; script exécutable par un tiers. Correspondances vers
le schéma Edamam isolées dans un dictionnaire unique. Filtrage par catégorie « champagne » plutôt
que par texte libre.

Résultat : 10 produits collectés, cinq champs renseignés ; composition manquante pour 2 produits.

Constats sur le contenu :

- 5 étiquettes de catégorie différentes pour 10 produits, dont des libellés non traduits
  (*fr:Champagnes bruts*) et un quasi vide de sens (*fr:Liquide*) ;
- un libellé mêlant caractères cyrilliques et fragments d'étiquette : saisie collaborative ;
- un cocktail à la pêche parmi les « champagnes ».

Limite et enseignement : la qualité et l'homogénéité des métadonnées externes sont un préalable à
tout réentraînement. Les catégories y sont, comme sur la marketplace, déclarées par des
contributeurs sans règle commune.

---

# 8. Limites et recommandations

Limites :

- **Étalon bruité** : catégories saisies par les vendeurs, dont l'étude cherche précisément à
  corriger les erreurs ; aucune performance rapportée n'échappe à cette référence imparfaite.
- **Volume** : 735 images d'entraînement. Extracteurs conservés figés : choix méthodologique
  limitant le surapprentissage, non impossibilité de principe ; explique aussi la
  contre-performance de Word2Vec, appris sur ce seul corpus.
- **Équilibre artificiel** : 7 × 150 exactement, jamais observé sur un catalogue réel.
- **Granularité** : premier niveau d'arborescence uniquement.
- **Indépendance historique du test** : consulté lors de la version erronée du protocole
  (partie 9) avant la correction ; le résultat final découle, lui, d'une sélection sur validation.

Recommandations :

- Faire trancher par l'équipe catalogue les frontières ambiguës ; ré-étiqueter un échantillon des
  produits où le modèle contredit le vendeur avec confiance.
- Mettre en service la fusion (partie 6) avec un seuil de confiance : cas incertains en revue
  humaine.
- Sur corpus élargi : réglage fin des extracteurs, classification hiérarchique.
- Data augmentation par type de produit plutôt qu'uniforme.
- Épicerie fine : traiter la qualité des métadonnées externes avant le modèle.

---

# 9. Conduite de projet

| Élément | Dispositif |
|---|---|
| Reproductibilité | graine fixe · versions épinglées · découpe unique (`src/pipeline.py`) |
| Validation | train / validation / test stratifié · sélection sur validation |
| Qualité | tests unitaires · CI depuis un clone vierge |
| Performance | cache disque des caractéristiques extraites |
| Coût | calcul local · ~20 min · aucun coût d'infrastructure |

Le poste dominant est le temps de compréhension des données et de choix des protocoles de mesure,
non l'écriture du code de modélisation. Incidents techniques : annexe C.

**Corrections méthodologiques.**

- Standardisation par dimension avant projection : supprimée. Elle ramenait l'accord TF-IDF à
  0,001, au niveau du hasard.
- Sélection des stratégies lue sur le jeu de test : corrigée par la sélection sur validation.

La correction du protocole de sélection a modifié la configuration retenue : la comparaison sur le
test désignait l'absence d'augmentation, celle sur la validation désigne l'augmentation douce.
Aucune des deux erreurs n'a été signalée par un test qui échoue : le code fonctionnait, il mesurait
autre chose que ce qui était visé. Chaque chiffre du rapport est pour cette raison accompagné de
son protocole, et certains sont rapportés deux fois.

**Enseignements.**

- Le choix de la représentation a un effet majeur : 0,044 à 0,510 en accord non supervisé, 0,8216 à
  0,9118 en F1 supervisé, sur les mêmes photographies.
- Une méthode reconnue peut être inadaptée sans être mauvaise : SIFT répond à une autre question.
- Le protocole de mesure fait partie du résultat : un score sans son protocole n'est pas une
  information.

---

# Annexes

## A. Rejeu de l'étude

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-encoders.txt

python faisabilite.py                   # les sept représentations, projections, K-means, ARI
python supervise_image.py               # classification supervisée et data augmentation
python collecte_api.py                  # collecte « champagne » et fichier CSV
python scripts/comparer_modernes.py     # le benchmark des représentations et la fusion
```

Le socle seul (`requirements.txt`) suffit pour l'exploration des données et les modèles classiques ;
`requirements-encoders.txt` ajoute les réseaux pré-entraînés, soit environ 3 Go de poids au premier
lancement. Les versions sont épinglées, la graine aléatoire est fixe, et la découpe des données est
définie dans un module unique (`src/pipeline.py`) appelé par tous les scripts. Les caractéristiques
extraites sont mises en cache sur disque : une seconde exécution ne recalcule ni SIFT ni les
réseaux.

## B. Figures et fichiers produits

| Fichier | Contenu |
|---|---|
| `reports/fig3_transformations.png` | La chaîne de transformation du texte, sur un article réel |
| `reports/fig4_donnees.png` | Équilibre des classes et distribution des longueurs |
| `reports/fig5_projections.png` | Les sept projections en deux dimensions |
| `reports/fig6_clusters.png` | VGG16 : catégories réelles et groupes trouvés |
| `reports/fig7_ari.png` | L'accord entre groupes et catégories, par représentation |
| `reports/fig8_confusion_image.png` | Matrice de confusion du modèle supervisé image |
| `reports/fig9_augmentation_par_classe.png` | L'effet de l'augmentation, catégorie par catégorie |
| `reports/fig10_comparaison_par_classe.png` | F1 par catégorie des six configurations du benchmark |
| `reports/faisabilite.csv` | Dimensions et accords des sept représentations |
| `reports/supervise_image_validation.csv` | Comparaison des stratégies d'augmentation |
| `reports/supervise_image_test.csv` | Résultat final du modèle image retenu |
| `reports/comparaison_validation.csv` | Le benchmark des représentations, sur la validation |
| `reports/comparaison_test.csv` | L'évaluation finale de la fusion, sur le jeu réservé |
| `reports/produits_champagne.csv` | Les dix produits collectés via l'API |

## C. Notes techniques

**Universal Sentence Encoder et la cohabitation TensorFlow / PyTorch.** USE est distribué pour
TensorFlow, quand le reste de la chaîne repose sur PyTorch. Chargées dans un même processus, les
deux bibliothèques se sont bloquées mutuellement sans lever d'erreur ; isolé dans son propre
processus, le modèle se charge en deux secondes. L'encodage USE est donc délégué à un sous-processus
dédié, ce qui permet d'employer le modèle de référence lui-même. Par ailleurs, `tensorflow_hub`
importe encore `pkg_resources`, retiré de `setuptools` à partir de la version 81 : la borne haute
des dépendances est là pour cette seule raison.

**Point d'entrée de l'API Open Food Facts.** L'ancien point d'entrée `cgi/search.pl`, déprécié, a
renvoyé une erreur de service lors du premier essai. Le script utilise le point d'entrée v2,
maintenu, et réessaie avec une attente croissante.

**Normalisation avant projection.** Chaque produit est ramené à une longueur unitaire avant l'ACP,
ce qui rend la distance euclidienne équivalente à la distance cosinus. La standardisation par
dimension, testée d'abord, amplifiait les termes rares au point de ramener l'accord TF-IDF au
niveau du hasard.
