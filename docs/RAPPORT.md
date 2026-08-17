# Automatisation de la catégorisation d'articles sur une place de marché

**Rapport de conduite de projet AI Engineering**

Richard Hugou · août 2026

---

# 1. Contexte et analyse des besoins

## 1.1 Contexte

| Élément | État constaté |
|---|---|
| Activité | place de marché généraliste anglophone, mise en relation vendeurs et acheteurs |
| Modèle de mise en ligne | dépôt libre par le vendeur : une photographie, une description, une catégorie déclarée |
| Volume | catalogue en phase de croissance ; échantillon de travail de 1 050 articles sur 7 catégories |
| Maturité IA et MLOps | nulle. Aucun modèle en service, aucun indicateur de qualité du catalogue |
| Moyens de calcul | poste de travail local, sans accélérateur dédié |

Enjeux : fluidifier la mise en ligne côté vendeur, fiabiliser le filtrage par catégorie côté
acheteur. Le second dépend du premier : un article mal rangé n'est pas retrouvé par l'acheteur qui
filtre, et le vendeur n'en est jamais informé. L'automatisation doit être disponible avant que le
volume rende la correction manuelle impraticable.

Contraintes de départ : calcul local sans accélérateur ; aucune donnée personnelle dans le
périmètre, mais des entrées non maîtrisées puisque fournies par les vendeurs ; solution devant tenir
sur un catalogue d'un ordre de grandeur supérieur à l'échantillon ; aucune plateforme d'hébergement
de modèle en place.

## 1.2 Besoin métier

Commanditaire : Lead Data Scientist. Utilisateurs concernés : vendeurs et équipe catalogue.

**Recueil.** Deux voies, sans atelier collectif :

1. **Brief écrit du commanditaire**, en trois demandes : établir la faisabilité à partir des données
   existantes, produire une classification supervisée à partir des images, éprouver la collecte de
   nouveaux produits via une interface externe.
2. **Analyse documentaire du jeu de données**, outillée : profilage des champs, des longueurs de
   description, des formats d'image, du taux de complétude (`scripts/profile_data.py`). Elle a fait
   remonter deux besoins absents du brief, la neutralisation du champ `brand` et l'harmonisation des
   formats d'image.

**Objectifs.**

| Nature | Objectif | Traduction mesurable |
|---|---|---|
| Business | réduire le temps de mise en ligne | part du catalogue classée sans intervention humaine |
| Business | réduire les articles mal rangés | taux de propositions correctes sur la part automatisée |
| Technique | établir que la catégorie est déductible des données existantes | accord entre groupes non supervisés et catégories réelles |
| Technique | évaluer le modèle sur des données jamais vues | F1 macro et exactitude sur un jeu réservé |

**Contraintes.**

| Type | Contrainte | Conséquence retenue |
|---|---|---|
| Fonctionnelle | premier niveau de nomenclature uniquement, 7 catégories | granularité fine hors périmètre |
| Fonctionnelle | proposition, jamais imposition, de la catégorie | sortie probabiliste, décision séparée du modèle |
| Non fonctionnelle | latence compatible avec une mise en ligne interactive | mesure du coût d'inférence par article |
| Non fonctionnelle | reproductibilité des essais | découpe unique, graine fixe, versions épinglées |
| Organisationnelle | aucune équipe d'annotation disponible | pas de ré-étiquetage du corpus |
| Réglementaire | corpus sans donnée personnelle, images sous licence de recherche | usage d'étude ; une extension à des données vendeurs relèverait du RGPD |
| Éthique | étiquette de référence produite par les vendeurs | la performance mesurée est bornée par la qualité de cette référence |

**Périmètre retenu**, ordonné par impact sur le besoin puis par effort de réalisation :
faisabilité avant tout engagement ; classification à partir du texte et de l'image ; décision
d'abstention et routage en revue humaine ; neutralisation des champs porteurs de fuite ; collecte
externe. Hors périmètre : classification hiérarchique complète, réglage
fin des extracteurs, interface de correction.

---

# 2. Audit de la solution existante

## 2.1 Processus en place

Aucune brique logicielle n'intervient entre le vendeur et le catalogue. La catégorie est saisie dans
un formulaire libre, enregistrée telle quelle, puis utilisée comme clé de filtrage.

![Le processus de catégorisation en place](../reports/fig11_flux_actuel.png)

| Composant | Nature | Outillage |
|---|---|---|
| Saisie | formulaire de mise en ligne | champ de nomenclature déclaratif |
| Stockage | catalogue produits | arborescence de catégories sur plusieurs niveaux |
| Restitution | recherche et filtres | filtrage exact sur le premier niveau |
| Contrôle qualité | aucun | ni règle partagée, ni revue, ni indicateur |

Données disponibles par article : `product_name`, `description` rédigée par le vendeur,
`product_category_tree`, une photographie, et une quinzaine de champs annexes hors périmètre.

**Exemple de référence, conservé dans tout le rapport.**

| Champ | Contenu |
|---|---|
| `product_name` | V9 METAL STRAP Analog Watch – For Men |
| `description` | *Specifications of V9 METAL STRAP Analog Watch – For Men. General Type Analog. Style Code METAL STRAP. Occasion Casual. Ideal For Men. Warranty NO. Body Features Dial Shape Round. Strap Color STEEL. Dial Color BLACK.* |
| `product_category_tree` | `["Watches >> Wrist Watches >> V9 Wrist Watches >> ..."]` |
| `image` | photographie de 1 152 × 1 816 pixels |

## 2.2 Audit des données et écarts

**Démarche.** Profilage descriptif de chaque champ, contrôle de la cible, contrôle des entrées.
Référentiel appliqué : complétude, validité, cohérence, représentativité, absence de fuite. L'audit
porte sur les données et sur le processus, aucun composant logiciel n'étant en place.

**Cible.** La catégorie n'est pas un champ, mais le premier niveau d'une chaîne d'arborescence mal
formée, aux échappements incohérents. Son extraction produit les sept catégories du périmètre :
*Baby Care*, *Beauty and Personal Care*, *Computers*, *Home Decor & Festive Needs*,
*Home Furnishing*, *Kitchen & Dining*, *Watches*. Elle est déclarée par les vendeurs : la référence
d'évaluation est elle-même faillible.

**Entrées.** Les descriptions sont des formulaires de spécifications aplatis, non des textes
rédigés : 13 à 587 mots, médiane 44, longueur médiane variable selon la catégorie, de 24 mots pour
*Home Furnishing* à 88 pour *Kitchen & Dining*. Les photographies présentent 890 tailles distinctes
sur 1 050 fichiers, des ratios de 0,23 à 4,36, jusqu'à 93 mégapixels : harmonisation obligatoire.

![Équilibre des classes et distribution des longueurs de description](../reports/fig4_donnees.png)

**Fuite détectée.** Le champ `brand` présente 32 % de valeurs manquantes, et cette absence est
fortement corrélée à la catégorie : 95 % des 338 absences se concentrent sur trois catégories,
*Watches* (140 / 150), *Beauty and Personal Care* (109 / 150) et *Kitchen & Dining* (71 / 150). Un
modèle recevant ce champ apprendrait une habitude de saisie, non le produit. Exclusion décidée,
indicateur d'absence compris.

**Écarts.** Le processus en place a été évalué sur la justesse du classement, la validation des
entrées, le coût et la maintenabilité.

| Écart | Conséquence |
|---|---|
| Aucune mesure de justesse du catalogue | aucun pilotage possible, le défaut reste invisible |
| Aucune règle commune entre vendeurs | deux articles semblables classés à deux endroits |
| Entrées libres non validées | texte et image non contrôlés en entrée de chaîne |

Aucun de ces écarts ne se corrige par un ajustement du formulaire.

---

# 3. Solution technique cible

## 3.1 Cas d'usage

Les cas d'usage sont dérivés du périmètre, puis cotés sur la valeur attendue, la faisabilité établie
par la mesure et l'effort. Un cas d'usage n'est retenu que si sa faisabilité est démontrée sur les
données du projet.

| Cas d'usage | Valeur | Faisabilité | Effort | Rang |
|---|---|---|---|---|
| **U1** Proposer la catégorie à la mise en ligne | forte | établie, partie 3.2 | moyen | **1** |
| **U2** S'abstenir sous un seuil et router en revue | forte | établie, partie 4.2 | faible | **2** |
| **U3** Signaler les articles dont l'étiquette contredit le modèle | moyenne | sous-produit de U1 | faible | **3** |
| **U4** Élargir la gamme par collecte externe | moyenne | établie mais dégradée, partie 3.5 | moyen | **4** |
| U5 Classer sur la nomenclature complète | moyenne | non établie, volume insuffisant | fort | écarté |
| U6 Rechercher des articles visuellement similaires | faible | non mesurée | moyen | écarté |

U1 à U3 partagent un même modèle : U2 et U3 se dérivent des probabilités produites par U1, sans coût
supplémentaire.

## 3.2 Comparatif des approches

**L'information est-elle présente dans les données existantes ?** Catégories masquées, huit
représentations projetées, les sept imposées par le brief et une variante en bigrammes,
partitionnement en sept groupes, accord mesuré par l'indice de Rand ajusté (annexe B).

VGG16 **0,510** · USE 0,440 · TF-IDF 0,325 · BERT 0,316 · Word2Vec 0,300 · SIFT 0,044.

![Les huit projections, couleur : catégorie réelle](../reports/fig5_projections.png)

Un indice de 0,51 n'est pas une proportion d'articles bien classés, mais une mesure de correspondance
entre deux partitions. Sur les mêmes photographies, un réseau convolutif atteint 0,510 quand SIFT
reste au niveau du hasard : une méthode reconnue peut répondre à une autre question que celle posée.
Côté texte, un comptage simple de mots fait jeu égal avec BERT, le vocabulaire discriminant et la
syntaxe presque absente.

**L'information nécessaire est présente dans les données déjà fournies par les vendeurs.** U1 est
faisable, l'image en étant la source la plus prometteuse en non supervisé. La confusion la plus
tenace, des textiles imprimés photographiés à plat mêlant *Home Furnishing* et *Baby Care*,
69 articles déplacés, traverse ensuite tout le projet : l'algorithme regroupe par matière et mise en
scène, la nomenclature par usage commercial.

**Quelle représentation retenir en supervisé ?** Découpe stratifiée figée des 1 050 articles en
735 entraînement, 157 validation, 158 test ; extracteurs pré-entraînés conservés figés ; une
architecture de classifieur unique, un perceptron à une couche cachée de 256 neurones, entraîné
indépendamment derrière chaque représentation. Un écart entre deux lignes ne peut donc provenir que
de la représentation. Sélection au F1 macro sur la validation ; le jeu de test n'intervient jamais
dans la sélection.

| Représentation | Modalité | Dimensions | F1 macro, validation |
|---|---|---|---|
| **TF-IDF** | texte | 4 532 | **0,937** |
| DINOv2 figé, Vision Transformer | image | 1 536 | 0,912 |
| BERT figé | texte | 768 | 0,905 |
| ModernBERT figé | texte | 768 | 0,904 |
| VGG16 figé, CNN | image | 512 | 0,822 |

Côté texte, les deux encodeurs figés ne devancent pas la référence lexicale. Côté image, DINOv2
dépasse VGG16 de neuf points, sur les mêmes photographies.

**Les deux modalités se complètent-elles ?** Les deux meilleures représentations par modalité sont
concaténées après normalisation ligne à ligne, soit 6 068 caractéristiques, même classifieur : la
fusion obtient **0,937** sur la validation, contre 0,937 pour TF-IDF seul. Les deux configurations
sont indiscernables à cette précision, aucun gain de la multimodalité n'est établi sur ce corpus. La
règle de sélection fixée d'avance désigne la fusion. La lecture par catégorie (annexe B) est plus
informative que la moyenne : la fusion gagne sur *Kitchen & Dining* et *Beauty and Personal Care*,
perd sur *Computers* et *Home Decor & Festive Needs*.

| Approche | Avantage | Inconvénient |
|---|---|---|
| TF-IDF seul | performance de tête, coût négligeable, interprétable | muet sur un article mal décrit |
| Encodeur de texte figé | robuste aux reformulations | ne devance pas TF-IDF ici, coût supérieur |
| VGG16 figé | rapide, éprouvé | neuf points sous DINOv2 |
| DINOv2 figé | meilleure représentation image, indépendante du texte | poste de calcul dominant |
| **Fusion TF-IDF ⊕ DINOv2** | **retenue par la règle ; les modalités ne faiblissent pas aux mêmes endroits** | **coût de DINOv2 sans gain moyen établi** |

## 3.3 Classification supervisée à partir des images seules

Demande distincte du brief, même protocole. Extracteur VGG16 figé, tête de classification apprise,
quatre stratégies d'augmentation comparées sur la validation (annexe B) : écart maximal de
0,006 point, insuffisant pour conclure à une amélioration nette. L'augmentation déplace les erreurs
plutôt qu'elle ne les supprime. Configuration retenue selon la règle fixée d'avance, augmentation
douce ×4, avantage non établi.

Évaluation finale, jeu de test, une seule ouverture : **137 / 158 articles correctement classés,
exactitude 86,7 %, F1 macro 0,867**, sans aucun recours au texte.

![Matrice de confusion du modèle image sur le jeu réservé](../reports/fig8_confusion_image.png)

La confusion *Baby Care* et *Home Furnishing* reproduit celle de l'étude non supervisée : ambiguïté
réelle entre certaines images, réduite par la supervision, non éliminée.

## 3.4 Architecture cible

![Architecture cible : service de catégorisation, décision par seuil, socle technique](../reports/fig12_architecture_cible.png)

**Évaluation finale de la solution retenue.** Jeu de test, une seule ouverture :
**F1 macro 0,987, 156 / 158 articles correctement classés**. Les deux erreurs restantes : un lit king
size étiqueté *Beauty and Personal Care*, lu *Home Furnishing*, étiquette probablement incohérente ;
un sticker mural lu *Baby Care*, ambiguïté d'univers domestique déjà observée en partie 3.2.

| Choix | Justification |
|---|---|
| Extracteurs figés, non réglés finement | 735 images d'entraînement ; le réglage fin relève d'une expérimentation dédiée |
| Perceptron à une couche de 256 | architecture constante entre les lignes comparées : c'est ce qui rend le comparatif interprétable |
| Fusion par concaténation, normalisation ligne à ligne | sans état à ajuster entre articles, donc sans fuite possible |
| Sérialisation en artefact unique de 37,5 Mo | transportable, chargeable sans base de modèles |
| Interface applicative conteneurisée | déploiement reproductible, indépendant de la machine hôte |
| Décision par seuil, séparée du modèle | paramètre métier révisable sans réentraînement |
| Journalisation et surveillance de dérive | proposées, sans objet avant une mise en service |

**Adéquation aux besoins.**

| Axe | Constat | Verdict |
|---|---|---|
| Fonctionnalités | U1 à U3 couverts par un modèle unique | conforme |
| Performances | F1 macro 0,987, 156 / 158 sur jeu réservé | conforme |
| Réglementaire | aucune donnée personnelle, champ `brand` écarté | à réexaminer avant extension |
| Sécurité | entrées libres à valider, interface à authentifier | disposition à mettre en place |
| Scalabilité | 1,62 h de calcul pour 100 000 articles | conforme, traitement par lots |
| Coûts | extraction d'image à 99,8 % du coût d'inférence | levier identifié |

| Poste | Coût mesuré par article |
|---|---|
| Extraction DINOv2 | 58,09 ms |
| Vectorisation TF-IDF | 0,10 ms |
| Tête de classification | 0,02 ms |
| **Total** | **58,22 ms** |

Renoncer à la modalité image ramènerait l'inférence à 0,12 ms par article, pour une F1 macro de
validation inchangée à cette précision.

## 3.5 Collecte externe

Cas d'usage U4, éprouvé sur l'épicerie fine. Source retenue : Open Food Facts, sans inscription,
donc rejouable par un tiers. Correspondances vers le schéma cible isolées dans un dictionnaire
unique, filtrage par catégorie plutôt que par texte libre. Résultat : 10 produits collectés, cinq
champs renseignés, composition manquante pour 2 produits.

Le contenu est hétérogène : 5 étiquettes de catégorie différentes pour 10 produits, des libellés non
traduits (*fr:Champagnes bruts*), un quasi vide de sens (*fr:Liquide*), un libellé mêlant caractères
cyrilliques et fragments d'étiquette, et un cocktail à la pêche parmi les « champagnes ». La
faisabilité technique est établie, la qualité des métadonnées ne l'est pas : les catégories y sont,
comme sur la place de marché, déclarées par des contributeurs sans règle commune.

---

# 4. Stratégie de mise en œuvre

## 4.1 Démarche projet

Découpage en lots courts, chacun sur sa propre branche, fusionné après revue, puis étiqueté. Les
jalons sont les versions réellement publiées du dépôt.

| Phase | Contenu | Outils | Jalon | État |
|---|---|---|---|---|
| 1. Audit des données | profilage, contrôle de la cible, détection de fuite | pandas, matplotlib | `v0.1.0` | réalisé |
| 2. Faisabilité | huit représentations, projection, partitionnement, accord | scikit-learn, PyTorch, TensorFlow Hub | `v0.1.0` | réalisé |
| 3. Comparatif supervisé | protocole constant, cinq représentations, fusion | scikit-learn, transformers | `v1.0.0` | réalisé |
| 4. Qualité et rejeu | tests unitaires, test anti-fuite, intégration continue | pytest, ruff, GitHub Actions | `v1.0.1` | réalisé |
| 5. Démonstrateur | interface de démonstration, trois modalités comparées | Streamlit, Docker | `v1.1.0` | réalisé |
| 6. Documentation | rapport, carnets exécutés, README | Markdown, nbformat | `v2.0.0` | réalisé |
| 7. Mise en service | interface authentifiée, seuil paramétrable | FastAPI, Docker | à planifier | proposé |
| 8. Surveillance | journalisation, suivi du taux d'automatisation, dérive | journaux applicatifs | à planifier | proposé |
| 9. Réentraînement | corpus ré-étiqueté, rejeu du protocole | chaîne existante, inchangée | à planifier | proposé |

Les phases 1 à 6 ont été conduites par une seule personne, de la conception à la rédaction. Les
phases 7 à 9 supposent une intégration dans le système de la place de marché et une équipe pour la
revue des cas incertains.

## 4.2 Aide à la prise de décision

**Indicateurs, définis puis évalués.** Le seuil de décision est choisi sur la validation, selon une
exigence posée d'avance : au moins 99 % de propositions correctes sur la part automatisée. Le seuil
le plus bas qui la satisfait est 0,60. Le couple d'indicateurs est ensuite mesuré une seule fois sur
le jeu de test.

| Indicateur | Nature | Cible | Validation | Test |
|---|---|---|---|---|
| Taux d'automatisation | business | maximiser | 84,1 % | **85,4 %** |
| Propositions correctes sur la part automatisée | business | ≥ 99 % | 0,992 | **0,993** |
| Volume en revue humaine | business | minimiser | 25 / 157 | **23 / 158** |
| F1 macro | technique | ≥ 0,90 | 0,937 | **0,987** |
| Latence d'inférence | technique | compatible interactif | non applicable | **58,22 ms** |
| Rejeu complet de la chaîne | technique | une commande | vérifié en intégration continue | vérifié |

Une seule erreur subsiste dans la part automatisée du jeu de test, sur 135 articles traités sans
intervention. Le seuil transforme un modèle à deux erreurs en un service dont l'erreur résiduelle
automatisée est unitaire, au prix de 14,6 % du volume envoyé en revue.

**Risques et mesures.**

| Risque | Impact | Mesure |
|---|---|---|
| Étiquettes vendeurs bruitées | performance biaisée, plafond de mesure | revue d'un échantillon par l'équipe catalogue, en priorité les contradictions confiantes (U3) |
| Distribution artificielle, 7 × 150 | généralisation inconnue | réévaluation sur un extrait réel du catalogue |
| Photographies de catalogue soignées | baisse probable en production | mesure sur un lot de prises de vue vendeurs |
| Coût de DINOv2 | latence et coût d'exploitation | traitement par lots, cache, ou repli sur le texte seul |
| Dérive du catalogue | perte de justesse silencieuse | suivi du taux d'automatisation, seuil d'alerte |
| Entrées vendeurs non maîtrisées | sécurité | validation des formats et des tailles en entrée |
| Charge de revue non affectée | 14,6 % du volume sans destinataire | dimensionner la file avant ouverture du service |
| Images sous licence de recherche | réutilisation en production exclue | constituer un corpus propre à l'entreprise |

Deux propriétés jouent en sens inverse : le seuil est un paramètre métier, donc l'arbitrage
automatisation contre justesse se révise sans réentraînement ; et la modalité image est séparable,
ce qui ouvre deux niveaux de service, économique ou complet.

**Scénarios.** Les charges n'ont pas été estimées dans le cadre du projet.

| Scénario | Périmètre | Charge |
|---|---|---|
| Démonstrateur | état actuel | réalisé |
| Mise en service | interface authentifiée, seuil paramétrable, journalisation | à estimer |
| Industrialisation | surveillance, réentraînement, interface de revue | à estimer |

---

# 5. Contrôle et suivi du projet

## 5.1 Tableau de bord

| Indicateur | Source | Valeur de fin de projet |
|---|---|---|
| Avancement | jalons publiés du dépôt | 6 phases sur 6 du périmètre, 7 versions étiquetées |
| Délais | historique du dépôt | 4 lots livrés, chacun sur sa branche puis fusionné |
| Livrables | dépôt et documentation | 3 scripts d'exécution, 6 carnets exécutés, rapport, démonstrateur en ligne |
| Qualité des données | profilage et contrôles | 1 fuite détectée et neutralisée, 2 anomalies de format corrigées |
| Qualité logicielle | intégration continue | 23 tests, dont un test anti-fuite ; lint et format à chaque poussée |
| Performances | jeu réservé, une ouverture | F1 macro 0,987, 156 / 158 |
| Coût de calcul | mesure locale | chaîne complète rejouée en environ 20 minutes |

**Gestion.** Flux à trois niveaux : `main` porte les versions publiées, `develop` intègre, chaque lot
vit sur sa propre branche avant fusion. Versions sémantiques étiquetées à chaque livraison,
intégration continue déclenchée sur les trois niveaux. Un correctif urgent a suivi le chemin dédié
(`hotfix/`), après détection d'un défaut de collecte des tests depuis un clone vierge.

## 5.2 Tests et suivi

| Niveau | Objet | Dispositif |
|---|---|---|
| Unitaire | découpe, métriques, fusion, vectorisation | 23 tests, exécutés à chaque poussée |
| Intégrité du protocole | les trois parts sont disjointes | test dédié, échoue si une fuite apparaît |
| Reproductibilité | la chaîne se rejoue depuis un clone vierge | tâche d'intégration continue distincte |
| Non-régression | le modèle rechargé reproduit le résultat publié | vérification à la sérialisation, artefact supprimé si écart |
| Bout en bout | interface sur articles jamais vus | démonstrateur conteneurisé, en ligne |

**Deux erreurs de méthode corrigées en cours de projet**, aucune n'ayant été signalée par un test en
échec : le code fonctionnait, il mesurait autre chose que ce qui était visé.

| Erreur | Correction | Conséquence |
|---|---|---|
| Comparaison des stratégies d'augmentation lue sur le jeu de test | sélection reportée sur la validation | configuration retenue modifiée |
| Standardisation par dimension avant projection, accord TF-IDF tombé à 0,001 | normalisation ligne à ligne | accord rétabli à 0,325 |

Chaque chiffre du rapport est pour cette raison accompagné de son protocole.

**Suivi en production, proposé** : taux de refus des propositions par les vendeurs, part du volume
au-dessus du seuil, distribution des confiances maximales pour la dérive des entrées, volume et délai
d'écoulement de la file de revue, latence et taux d'erreur de l'interface.

---

# 6. Conclusion et recommandations

L'information nécessaire à la catégorisation est présente dans les données déjà fournies par les
vendeurs : la faisabilité est établie avant tout engagement. Le comparatif à protocole constant
désigne la fusion TF-IDF ⊕ DINOv2, qui obtient 0,987 de F1 macro et 156 articles sur 158 sur un jeu
réservé ouvert une seule fois. La décision est séparée du modèle par un seuil de confiance choisi sur
la validation, qui automatise 85,4 % du volume avec 0,993 de propositions correctes. Les modèles
récents n'apportent pas systématiquement un gain : net en vision, nul ici sur le texte figé, où une
référence lexicale reste en tête.

**Recommandations, par ordre de priorité.** La première est la prochaine étape.

1. Faire arbitrer par l'équipe catalogue les frontières ambiguës, et ré-étiqueter en priorité les
   articles où le modèle contredit le vendeur avec une confiance élevée.
2. Mettre en service l'interface authentifiée avec seuil paramétrable et journalisation des
   prédictions.
3. Dimensionner la file de revue humaine avant l'ouverture du service, 14,6 % du volume étant
   concerné.
4. Réévaluer la performance sur un extrait réel du catalogue, à distribution non équilibrée et
   photographies non retouchées.
5. Constituer un corpus propre à l'entreprise, les images d'étude étant sous licence de recherche.
6. Traiter la qualité des métadonnées externes avant d'envisager l'élargissement de gamme.

**Perspectives.** Réglage fin des extracteurs sur corpus élargi ; classification sur la nomenclature
complète ; augmentation différenciée par type de produit ; recherche d'articles visuellement
similaires, qui réutiliserait les représentations déjà calculées.

---

# Annexes

## A. Rejeu de l'étude

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-encoders.txt

python faisabilite.py                   # les huit représentations, projections, partitionnement, accord
python supervise_image.py               # classification supervisée et augmentation
python collecte_api.py                  # collecte « champagne » et fichier CSV
python scripts/comparer_modernes.py     # le comparatif des représentations et la fusion
python scripts/seuil_confiance.py       # seuil de décision et indicateurs d'automatisation
python scripts/cout_inference.py        # coût d'inférence, poste par poste
python scripts/schemas.py               # les deux schémas de flux et d'architecture
```

Le socle seul (`requirements.txt`) suffit pour l'exploration des données et les modèles classiques ;
`requirements-encoders.txt` ajoute les réseaux pré-entraînés, soit environ 3 Go de poids au premier
lancement. Les versions sont épinglées, la graine aléatoire est fixe, et la découpe des données est
définie dans un module unique (`src/pipeline.py`) appelé par tous les scripts. Les caractéristiques
extraites sont mises en cache : une seconde exécution ne recalcule ni SIFT ni les réseaux.

## B. Tableaux détaillés

**Accord entre groupes non supervisés et catégories réelles.** Mesure rapportée deux fois, avant et
après réduction, la réduction étant déformante.

| Représentation | Modalité | Dimensions | Accord, projection | Accord, espace complet |
|---|---|---|---|---|
| **CNN (VGG16)** | image | 512 | **0,510** | **0,540** |
| USE | texte | 512 | 0,440 | 0,333 |
| TF-IDF | texte | 5 000 | 0,325 | 0,214 |
| Comptage + bigrammes | texte | 5 000 | 0,316 | 0,227 |
| BERT | texte | 768 | 0,316 | 0,288 |
| Comptage de mots | texte | 2 444 | 0,306 | 0,270 |
| Word2Vec | texte | 300 | 0,300 | 0,207 |
| SIFT + BoVW | image | 256 | 0,044 | 0,056 |

Correspondance de un à un entre groupes et catégories pour VGG16 : informatique 87 %, montres 86 %,
beauté 80 %. VGG16 est la seule représentation meilleure avant réduction qu'après, la séparation
existant déjà dans l'espace d'origine.

**Stratégies d'augmentation, sur la validation.**

| Stratégie | Images d'entraînement | F1 macro |
|---|---|---|
| Augmentation douce ×4 | 3 675 | **0,828** |
| Augmentation forte ×4 | 3 675 | 0,827 |
| Sans augmentation | 735 | 0,822 |
| Augmentation forte ×8 | 6 615 | 0,815 |

## C. Figures et fichiers produits

| Fichier | Contenu |
|---|---|
| `reports/fig3_transformations.png` | La chaîne de transformation du texte, sur un article réel |
| `reports/fig4_donnees.png` | Équilibre des classes et distribution des longueurs |
| `reports/fig5_projections.png` | Les huit projections en deux dimensions |
| `reports/fig6_clusters.png` | VGG16 : catégories réelles et groupes trouvés |
| `reports/fig7_ari.png` | L'accord entre groupes et catégories, par représentation |
| `reports/fig8_confusion_image.png` | Matrice de confusion du modèle supervisé image |
| `reports/fig9_augmentation_par_classe.png` | L'effet de l'augmentation, catégorie par catégorie |
| `reports/fig10_comparaison_par_classe.png` | F1 par catégorie des six configurations comparées |
| `reports/fig11_flux_actuel.png` | Le processus de catégorisation en place |
| `reports/fig12_architecture_cible.png` | L'architecture cible et son socle technique |
| `reports/faisabilite.csv` | Dimensions et accords des huit représentations |
| `reports/supervise_image_validation.csv` | Comparaison des stratégies d'augmentation |
| `reports/comparaison_validation.csv` | Le comparatif des représentations, sur la validation |
| `reports/comparaison_test.csv` | L'évaluation finale de la fusion, sur le jeu réservé |
| `reports/seuil_confiance.csv` | Automatisation et justesse par seuil, validation et test |
| `reports/cout_inference.json` | Coût d'inférence par poste et par volume |
| `reports/produits_champagne.csv` | Les dix produits collectés via l'interface externe |

## D. Ressources du projet

| Ressource | Adresse |
|---|---|
| Dépôt du projet | https://github.com/richardhugou/product-category-classifier |
| Démonstrateur en ligne | https://huggingface.co/spaces/trikwi/projet55 |
| Portfolio | https://portfolio.richardh.fr |
| Description du dépôt | `README.md` |
| Carnets exécutés, dans l'ordre de la mission | `notebooks/` |

## E. Notes techniques

**Universal Sentence Encoder et la cohabitation TensorFlow et PyTorch.** USE est distribué pour
TensorFlow, quand le reste de la chaîne repose sur PyTorch. Chargées dans un même processus, les deux
bibliothèques se sont bloquées mutuellement sans lever d'erreur ; isolé dans son propre processus, le
modèle se charge en deux secondes. L'encodage USE est donc délégué à un sous-processus dédié. Par
ailleurs, `tensorflow_hub` importe encore `pkg_resources`, retiré de `setuptools` à partir de la
version 81 : la borne haute des dépendances est là pour cette seule raison.

**Point d'entrée de l'interface Open Food Facts.** L'ancien point d'entrée `cgi/search.pl`, déprécié,
a renvoyé une erreur de service lors du premier essai. Le script utilise le point d'entrée v2,
maintenu, et réessaie avec une attente croissante.

**Mesure du coût d'inférence.** Latences mesurées sur poste local, accélérateur intégré, sur un
échantillon de 16 photographies pour l'extraction et sur les 158 articles du jeu réservé pour le
texte et la tête de classification. Un passage à blanc précède la mesure pour ne pas compter
l'allocation initiale.
