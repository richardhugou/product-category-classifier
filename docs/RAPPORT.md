# Automatisation de la catégorisation d'articles sur une place de marché

**Rapport de conduite de projet AI Engineering**

Richard Hugou · août 2026

---

# 1. Contexte et besoin

## 1.1 Contexte

La place de marché étudiée laisse au vendeur le choix de la catégorie lors de la mise en ligne d'un article. Les informations disponibles sont une description, une photographie et plusieurs champs déclaratifs.

Le catalogue est en croissance. Aucun modèle de catégorisation ni dispositif de contrôle qualité n'est en place. Le projet est réalisé sur poste local, à partir d'un échantillon de **1 050 articles répartis en 7 catégories**.

**Enjeu principal :** proposer automatiquement une catégorie à partir des informations déjà fournies par le vendeur.

Contraintes retenues :

- premier niveau de nomenclature uniquement ;
- calcul local ;
- solution reproductible et transportable ;
- proposition de catégorie, sans décision imposée au vendeur ;
- corpus d'étude sans donnée personnelle ;
- images sous licence de recherche.

## 1.2 Besoin métier

Le brief du commanditaire comporte trois demandes :

1. vérifier que la catégorie peut être retrouvée à partir du texte et de l'image ;
2. construire et évaluer une classification supervisée à partir des images ;
3. tester la collecte de nouveaux produits via une source externe.

L'analyse initiale des données ajoute deux exigences :

- exclure les variables susceptibles de créer une fuite ;
- harmoniser les formats d'image.

| Objectif | Indicateur |
|---|---|
| Établir la faisabilité | accord entre groupes non supervisés et catégories |
| Comparer les solutions | F1 macro sur validation |
| Évaluer le modèle retenu | F1 macro et exactitude sur test |
| Limiter les décisions incertaines | taux d'automatisation et justesse après seuil |
| Garantir le rejeu | pipeline exécutable depuis un dépôt vierge |

**Périmètre retenu :** catégorisation sur 7 classes, comparaison texte/image, fusion multimodale, seuil d'abstention et collecte externe.

**Hors périmètre :** classification hiérarchique complète, fine-tuning des extracteurs, interface de correction catalogue.

---

# 2. Audit de l'existant

## 2.1 Processus actuel

La catégorie est saisie par le vendeur puis utilisée directement pour le filtrage du catalogue. Aucun contrôle intermédiaire n'est prévu.

![Processus de catégorisation actuel](../reports/fig11_flux_actuel.png)

| Étape | Situation |
|---|---|
| Saisie | catégorie déclarée par le vendeur |
| Stockage | arborescence de catégories |
| Restitution | filtrage sur le premier niveau |
| Contrôle qualité | aucun |

## 2.2 Données disponibles

Chaque article contient notamment :

- `product_name` ;
- `description` ;
- `product_category_tree` ;
- une photographie.

Exemple utilisé comme référence dans l'étude :

| Champ | Exemple |
|---|---|
| `product_name` | V9 METAL STRAP Analog Watch – For Men |
| `description` | fiche de spécifications en anglais |
| `product_category_tree` | `Watches >> Wrist Watches >> ...` |
| `image` | 1 152 × 1 816 px |

Le premier niveau de `product_category_tree` fournit les 7 classes du projet :
*Baby Care*, *Beauty and Personal Care*, *Computers*, *Home Decor & Festive Needs*, *Home Furnishing*, *Kitchen & Dining*, *Watches*.

La catégorie reste une **étiquette déclarative** : elle peut contenir des erreurs.

## 2.3 Audit des entrées

Les descriptions sont des fiches de spécifications plutôt que du texte rédigé :

- 13 à 587 mots ;
- médiane : 44 mots ;
- longueur médiane différente selon les catégories.

Les photographies sont hétérogènes :

- 890 tailles distinctes sur 1 050 fichiers ;
- ratios de 0,23 à 4,36 ;
- jusqu'à 93 mégapixels.

![Équilibre des classes et longueurs de description](../reports/fig4_donnees.png)

### Prétraitements

Texte : passage en minuscules, suppression de la ponctuation, découpage en mots, retrait des
mots-outils anglais et des mots de moins de trois lettres. Sur l'exemple de référence, 35 mots bruts
donnent 29 jetons.

Ni stemming ni lemmatisation. Les descriptions contiennent 462 jetons alphanumériques distincts, soit
974 occurrences, principalement des références de modèles (`cr540e`), souvent les termes les plus
discriminants d'une fiche. Les tronquer ou les ramener à une forme canonique détruirait cette
information. Les chiffres sont conservés pour la même raison.

Image : 224 pixels et normalisation ImageNet pour le réseau convolutif ; niveaux de gris, égalisation
d'histogramme et 256 pixels pour SIFT.

### Fuite détectée

Le champ `brand` contient **32 % de valeurs manquantes**. Ces absences sont concentrées sur trois classes :

- *Watches* : 140 / 150 ;
- *Beauty and Personal Care* : 109 / 150 ;
- *Kitchen & Dining* : 71 / 150.

Le champ est exclu, indicateur d'absence compris. Son utilisation aurait permis d'apprendre un comportement de saisie plutôt que le contenu du produit.

### Écarts principaux

| Écart | Conséquence |
|---|---|
| aucune mesure de justesse | qualité du catalogue non pilotée |
| nomenclature appliquée librement | incohérences entre vendeurs |
| texte et image non contrôlés | entrées hétérogènes |
| étiquettes déclaratives | référence d'évaluation imparfaite |

---

# 3. Solution technique

## 3.1 Cas d'usage retenus

Priorisation selon valeur métier, faisabilité mesurée et effort de mise en œuvre.

| Cas d'usage | Décision |
|---|---|
| proposer une catégorie à la mise en ligne | retenu |
| s'abstenir sous un seuil de confiance | retenu |
| signaler les contradictions modèle / vendeur | retenu |
| collecter de nouveaux produits | étudié séparément |
| classifier toute la nomenclature | reporté |
| recherche visuelle de produits similaires | hors périmètre |

Les trois premiers cas d'usage reposent sur le même modèle.

## 3.2 Faisabilité sur les représentations

La première étape vérifie si les catégories sont déjà structurées dans les données, sans apprentissage supervisé.

Huit représentations sont comparées :

- texte : comptage, bigrammes, TF-IDF, Word2Vec, BERT, USE ;
- image : SIFT + BoVW, VGG16.

Réduction : ACP à 50 composantes, puis t-SNE pour la visualisation.

Après réduction et partitionnement en 7 groupes, l'accord avec les catégories est mesuré par l'**indice de Rand ajusté (ARI)**. L'ARI est également calculé dans l'espace de représentation original afin de contrôler l'effet de la réduction.

| Représentation | Modalité | ARI projection | ARI espace complet |
|---|---|---:|---:|
| **VGG16** | image | **0,510** | **0,540** |
| USE | texte | 0,440 | 0,333 |
| TF-IDF | texte | 0,325 | 0,214 |
| Comptage + bigrammes | texte | 0,316 | 0,227 |
| BERT | texte | 0,316 | 0,288 |
| Comptage de mots | texte | 0,306 | 0,270 |
| Word2Vec | texte | 0,300 | 0,207 |
| SIFT + BoVW | image | 0,044 | 0,056 |

![Projections des représentations](../reports/fig5_projections.png)

**Constat :** l'information de catégorie est présente dans les données. VGG16 produit la structure la plus nette en image ; USE est la meilleure représentation texte de cette étape non supervisée.

La confusion la plus visible concerne *Home Furnishing* et *Baby Care*, notamment pour des textiles photographiés dans des conditions proches.

## 3.3 Benchmark supervisé

Le benchmark compare les représentations dans un protocole commun :

- **735** articles d'entraînement ;
- **157** de validation ;
- **158** de test ;
- extracteurs pré-entraînés figés ;
- MLP à une couche cachée de 256 neurones ;
- une instance de MLP entraînée indépendamment pour chaque représentation ;
- sélection au **F1 macro sur validation**.

| Représentation | Modalité | F1 macro validation |
|---|---|---:|
| **TF-IDF** | texte | **0,937** |
| DINOv2 | image | 0,912 |
| BERT | texte | 0,905 |
| ModernBERT | texte | 0,904 |
| VGG16 | image | 0,822 |

**Classifieur.** Perceptron multicouche à une couche cachée de 256 neurones, activation ReLU, sortie
softmax sur 7 catégories. Optimiseur Adam, régularisation L2, 500 itérations au plus, arrêt anticipé
sur 10 % du jeu d'entraînement que le classifieur met de côté, graine fixe. Seule cette tête apprend :
les extracteurs restent figés.

**Texte :** BERT et ModernBERT n'apportent pas d'avantage par rapport à TF-IDF dans ce protocole.

**Image :** DINOv2 améliore nettement les performances par rapport à VGG16, avec environ **+9 points de F1 macro**.

![F1 par catégorie sur la validation](../reports/fig10_comparaison_par_classe.png)

### Fusion multimodale

Les deux meilleures représentations par modalité sont concaténées :

- TF-IDF : 4 532 caractéristiques ;
- DINOv2 : 1 536 caractéristiques ;
- représentation finale : 6 068 caractéristiques.

| Configuration | F1 macro validation |
|---|---:|
| TF-IDF | 0,937 |
| TF-IDF ⊕ DINOv2 | **0,937** |

Les deux résultats sont **quasi identiques**. Aucun gain moyen de la fusion n'est établi sur la validation.

La fusion est retenue par la règle de sélection fixée pour le benchmark, puis évaluée sur le jeu de test :

- **F1 macro : 0,987**
- **exactitude : 156 / 158**
- **2 erreurs**

Le score test est rapporté comme résultat sur cet échantillon de 158 articles, sans extrapolation à un catalogue réel.

## 3.4 Classification image et data augmentation

La demande image seule est traitée séparément avec VGG16 figé.

Quatre configurations sont comparées sur la validation :

| Stratégie | F1 macro validation |
|---|---:|
| augmentation douce ×4 | **0,828** |
| augmentation forte ×4 | 0,827 |
| sans augmentation | 0,822 |
| augmentation forte ×8 | 0,815 |

L'écart maximal est de **0,006**. Aucun bénéfice général de la data augmentation n'est établi.

Sur le protocole corrigé, la configuration retenue est évaluée sur test :

- **F1 macro : 0,867**
- **exactitude : 137 / 158**

![Matrice de confusion du modèle image](../reports/fig8_confusion_image.png)

Les confusions *Baby Care* / *Home Furnishing* observées en non supervisé restent visibles en classification.

## 3.5 Architecture cible

![Architecture cible](../reports/fig12_architecture_cible.png)

Chaîne proposée :

`texte + image → prétraitements → TF-IDF + DINOv2 → fusion → MLP → score → seuil → catégorie ou revue`

Principaux choix :

| Choix | Motif |
|---|---|
| extracteurs figés | comparaison contrôlée des représentations |
| MLP 256 identique | isoler l'effet de la représentation |
| seuil séparé du modèle | décision métier modifiable sans réentraînement |
| artefact sérialisé | transport et rejeu simplifiés |
| conteneurisation | environnement reproductible |

### Coût d'inférence

| Poste | Temps par article |
|---|---:|
| DINOv2 | 58,09 ms |
| TF-IDF | 0,10 ms |
| MLP | 0,02 ms |
| **Total** | **58,22 ms** |

L'extraction DINOv2 représente l'essentiel du coût. TF-IDF seul conserve un F1 de validation équivalent à la précision affichée pour un coût très inférieur.

## 3.6 Collecte externe

Open Food Facts est utilisé pour tester une collecte sur l'épicerie fine.

Résultat :

- 10 produits ;
- 5 champs collectés ;
- 2 compositions manquantes.

Les catégories obtenues sont hétérogènes, parfois non traduites ou peu informatives. La collecte est techniquement réalisable, mais la qualité des métadonnées doit être traitée avant tout usage d'apprentissage.

---

# 4. Mise en œuvre et aide à la décision

## 4.1 Démarche projet

Le projet est découpé en lots courts, versionnés dans Git.

| Phase | Contenu | État |
|---|---|---|
| Audit | profilage, cible, fuite | réalisé |
| Faisabilité | représentations, projection, ARI | réalisé |
| Benchmark | cinq représentations, fusion | réalisé |
| Qualité | tests, CI, anti-fuite | réalisé |
| Démonstrateur | interface trois modalités | réalisé |
| Documentation | README, rapport, notebooks | réalisé |
| Mise en service | authentification, seuil, logs | proposé |
| Surveillance | dérive et taux d'automatisation | proposé |
| Réentraînement | corpus ré-étiqueté | proposé |

Outils principaux : pandas, scikit-learn, PyTorch, Transformers, pytest, GitHub Actions, Streamlit, Docker.

## 4.2 Seuil de décision

Le modèle ne doit pas nécessairement décider pour tous les produits.

Le seuil est choisi sur la validation avec une cible de **99 % de propositions correctes sur la part automatisée**.

Seuil retenu : **0,60**.

| Indicateur | Validation | Test |
|---|---:|---:|
| taux d'automatisation | 84,1 % | **85,4 %** |
| justesse sur la part automatisée | 99,2 % | **99,3 %** |
| revue humaine | 25 / 157 | **23 / 158** |

Le modèle automatise donc environ **85 %** du volume et transfère le reste en revue humaine.

## 4.3 Risques

| Risque | Mesure proposée |
|---|---|
| étiquettes vendeurs bruitées | revue d'un échantillon par l'équipe catalogue |
| corpus équilibré artificiellement | nouvelle évaluation sur catalogue réel |
| photographies trop propres | test sur photographies vendeurs |
| coût de DINOv2 | batch, cache ou mode texte seul |
| dérive du catalogue | suivi du taux d'automatisation et des confiances |
| entrées libres | validation des formats et tailles |
| charge de revue humaine | dimensionnement avant mise en service |
| licence des images d'étude | constitution d'un corpus propre à l'entreprise |

### Estimation simple

Hypothèses : catalogue de 100 000 articles ; revue manuelle de 30 s/article ; taux d'automatisation
observé de 85,4 % ; temps d'inférence mesuré de 58,22 ms/article.

| Indicateur | Sans automatisation | Avec le modèle |
|---|---|---|
| Articles traités | 100 000 | 100 000 |
| Catégories automatisées | 0 | 85 400 |
| Articles en revue | 100 000 | 14 600 |
| Charge de revue à 30 s/article | ≈ 104 j-personne | ≈ 15 j-personne |
| Calcul modèle | aucun | ≈ 1,62 h |

Ordre de grandeur : environ 89 jours-personne de revue évités pour 100 000 articles.

### Mise en œuvre

Hypothèse : 1 Data Scientist + 1 profil MLOps à 600 €/jour/personne.

| Scénario | Charge | Estimation |
|---|---|---|
| Mise en service minimale | 2 × 5 jours | 6 000 € |
| Industrialisation | 2 × 10 jours | 12 000 € |

Estimations indicatives, hors infrastructure et maintenance.

---

# 5. Contrôle et suivi

## 5.1 Pilotage

| Indicateur | État final |
|---|---|
| périmètre | 6 phases réalisées |
| versions | 7 versions étiquetées |
| livrables | scripts, notebooks, rapport, démonstrateur |
| qualité des données | 1 fuite neutralisée, anomalies de format corrigées |
| qualité logicielle | 23 tests + lint + CI |
| performance finale | F1 macro 0,987 · 156 / 158 |
| coût de calcul | rejeu complet ≈ 20 min sur poste local |

Gestion Git :

- `main` : versions publiées ;
- `develop` : intégration ;
- branches dédiées par lot ;
- correctifs via `hotfix/`.

## 5.2 Tests et reproductibilité

| Niveau | Contrôle |
|---|---|
| unitaire | découpe, métriques, fusion, vectorisation |
| anti-fuite | train / validation / test disjoints |
| reproductibilité | rejeu depuis clone vierge |
| non-régression | artefact rechargé = résultat publié |
| bout en bout | démonstrateur conteneurisé |

Deux corrections méthodologiques ont été documentées :

1. une première comparaison des stratégies d'augmentation avait été lue sur le test ; la sélection a été déplacée sur la validation ;
2. une standardisation par dimension dégradait fortement l'ARI TF-IDF ; elle a été remplacée par une normalisation ligne à ligne.

Ces erreurs n'étaient pas des erreurs d'exécution : le code fonctionnait, mais le protocole devait être corrigé.

### Suivi proposé en production

- taux de refus des propositions ;
- part du volume automatisée ;
- distribution des scores de confiance ;
- volume et délai de la file de revue ;
- latence et taux d'erreur du service.

---

# 6. Conclusion et recommandations

La faisabilité de la catégorisation automatique est établie sur le corpus étudié.

Le benchmark montre deux résultats principaux :

- **texte :** TF-IDF reste devant BERT et ModernBERT dans le régime figé ;
- **image :** DINOv2 améliore nettement VGG16.

La fusion TF-IDF ⊕ DINOv2 ne montre pas de gain moyen sur la validation, mais constitue la configuration retenue par la règle de sélection du benchmark. Sur le test, elle atteint **0,987 de F1 macro et 156 / 158 classifications correctes**.

Le seuil de confiance permet de transformer le classifieur en service avec abstention : **85,4 %** du volume est automatisé sur le test, avec **99,3 %** de propositions correctes sur cette partie.

## Recommandations

1. Réévaluer la solution sur un échantillon réel du catalogue, non équilibré et avec des
   photographies vendeurs, et faire ré-étiqueter un échantillon des catégories ambiguës.
2. Mettre en service avec authentification, validation des entrées et journalisation, après avoir
   dimensionné la revue humaine.
3. Sur un corpus élargi, tester le fine-tuning de BERT / ModernBERT / DINOv2.
4. Répéter les expériences sur plusieurs splits ou seeds afin de quantifier l'incertitude, et
   réévaluer sur cette base l'intérêt de la fusion multimodale.
5. Étendre à la classification hiérarchique uniquement après validation sur le premier niveau.

---

# Annexes

## A. Rejeu

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-encoders.txt

python faisabilite.py
python supervise_image.py
python collecte_api.py
python scripts/comparer_modernes.py
python scripts/seuil_confiance.py
python scripts/rapport_classification.py
python scripts/cout_inference.py
python scripts/schemas.py
```

Versions épinglées, graine fixe, split centralisé dans `src/pipeline.py`, caractéristiques mises en cache.

## B. Figures

- `fig4_donnees.png` : équilibre des classes et longueurs ;
- `fig5_projections.png` : projections des représentations ;
- `fig8_confusion_image.png` : confusion image ;
- `fig10_comparaison_par_classe.png` : F1 par catégorie ;
- `fig11_flux_actuel.png` : processus existant ;
- `fig12_architecture_cible.png` : architecture cible.

## C. Ressources

- Dépôt : `github.com/richardhugou/product-category-classifier`
- Démonstrateur : `huggingface.co/spaces/trikwi/projet55`
- Portfolio : `portfolio.richardh.fr`
- Notebooks : `notebooks/`

## D. Précision et rappel par catégorie

Modèle retenu, jeu de test, `reports/classification_test.csv`.

| Catégorie | Précision | Rappel | F1 | Articles |
|---|---:|---:|---:|---:|
| Baby Care | 0,957 | 1,000 | 0,978 | 22 |
| Beauty and Personal Care | 1,000 | 0,955 | 0,977 | 22 |
| Computers | 1,000 | 1,000 | 1,000 | 23 |
| Home Decor & Festive Needs | 1,000 | 0,957 | 0,978 | 23 |
| Home Furnishing | 0,958 | 1,000 | 0,979 | 23 |
| Kitchen & Dining | 1,000 | 1,000 | 1,000 | 22 |
| Watches | 1,000 | 1,000 | 1,000 | 23 |
| **Moyenne macro** | **0,988** | **0,987** | **0,987** | **158** |

Valeurs exactes de la fusion sur validation, pour traçabilité : TF-IDF 0,9365, fusion 0,9366
(`reports/comparaison_validation.csv`).
