# Rapport de conduite de projet

**Catégorisation automatique d'articles de commerce en ligne**
Richard Hugou · août 2026

---

## Résumé exécutif

**Le problème.** Sur une place de marché, la catégorie d'un article est saisie par le vendeur. C'est
manuel, donc peu fiable, et le volume croît. Un article mal catégorisé est introuvable : invendu pour
le vendeur, invisible pour l'acheteur.

**Ce qui a été construit.** Un classifieur de fiches produit en sept catégories, à partir de la
description et de la photographie. Six approches comparées à protocole constant, une chaîne
reproductible en une commande, et une application de démonstration.

**Le protocole.** Découpe 70 / 15 / 15 stratifiée à graine fixe, 158 articles de test jamais vus,
seuil d'acceptation métier fixé **avant** toute mesure à F1 macro ≥ 0,90.

**Le résultat, en trois nombres.** F1 macro de **0,974** pour le modèle retenu, la fusion texte +
image. **0,955** sur la catégorie la plus faible, contre 0,870 pour le texte seul : l'image répare
précisément les classes que le texte confond. **82,9 %** du volume catégorisable sans aucune erreur
observée, au seuil de confiance de 0,60.

**La limite majeure.** Les encodeurs pré-entraînés sont évalués **figés**, sans réglage fin. Le
rapport compare des représentations, pas des capacités de modèles. Cette limite est développée
en §4.3.

**Recommandation n° 1.** Versionner les données et figer un échantillon dans le dépôt avant toute
reprise. L'étude initiale de 2024 n'était plus rejouable au moment de cette reprise, faute de
conservation des jeux intermédiaires — c'est le constat bloquant de l'audit.

---

## 1. Contexte et analyse des besoins

### 1.1 Le contexte

Une place de marché en ligne met en relation des vendeurs indépendants et des acheteurs. Chaque
article est publié avec une photographie et une description libre en anglais, et son vendeur lui
attribue lui-même une catégorie parmi sept.

| Dimension | État |
|---|---|
| Volume actuel | Faible — le catalogue est encore réduit |
| Trajectoire | Passage à l'échelle anticipé : c'est ce qui rend l'automatisation nécessaire |
| Maturité en apprentissage automatique | Nulle en production. Une étude de faisabilité existe, aucune brique n'est déployée |
| Contrainte de données | Aucune donnée personnelle dans le périmètre |
| Contrainte de coût | Le coût d'inférence doit rester proportionné à la valeur d'un article |
| Contrainte d'expérience | Le parcours de publication ne doit pas être ralenti |

L'enjeu n'est pas la performance d'un modèle : c'est de savoir **si une automatisation est possible,
à quel coût, et à partir de quel niveau de confiance on peut se passer d'un humain.**

### 1.2 La méthode de recueil des besoins

Quatre sources triangulées. Chaque besoin porte la trace de son origine ; ceux qui ne découlent
d'aucune source vérifiable sont marqués comme hypothèses de l'auteur.

| Source | Instrument | Production | Biais assumé |
|---|---|---|---|
| **S1 — Brief du commanditaire** | Demande écrite de la responsable data : périmètre, livrables, méthodes imposées | 5 besoins fonctionnels, 2 contraintes de méthode | Exprime autant la solution que le besoin — à requalifier systématiquement |
| **S2 — Analyse documentaire** | Cadre de protection des données, conditions d'usage du jeu | 1 besoin réglementaire | Énonce le minimum légal, pas le besoin d'usage |
| **S3 — Benchmark** | Fonctions de catégorisation observées sur des places de marché établies | 3 besoins non fonctionnels, le standard de marché | Les communications commerciales surestiment les capacités |
| **S4 — Personas et parcours** | Trois parcours reconstruits : vendeur, acheteur, responsable data | 3 besoins organisationnels | Construit par l'auteur : sert à **structurer**, jamais à **prioriser** |

Le commanditaire est documenté mais l'exercice reste une mise en situation. La légitimité des besoins
vient de la triangulation, pas d'une autorité unique.

### 1.3 Les personas

| Persona | Objectif | Geste actuel | Irritant | Critère de confiance |
|---|---|---|---|---|
| **Vendeur** | Publier vite | Choisit sa catégorie dans une liste longue | Se trompe ; l'article devient invisible | Une suggestion qu'il peut corriger |
| **Acheteur** | Trouver l'article | Navigue par catégorie et filtres | Résultats incohérents, articles absents | Le catalogue est rangé |
| **Responsable data** | Décider d'industrialiser | Attend une étude de faisabilité | Un score sans coût ni condition d'emploi | Une recommandation chiffrée |

### 1.4 Registre des besoins

`F` fonctionnel · `NF` non fonctionnel · `O` organisationnel · `T` technique · `R` réglementaire

| ID | Besoin | Type | Source | Priorité |
|---|---|---|---|---|
| REQ-01 | Attribuer une catégorie à partir de la description | F | S1 | **Must** |
| REQ-02 | Couvrir les 7 catégories du catalogue | F | S1 | **Must** |
| REQ-03 | Exposer un score de confiance avec la prédiction | F | S4 | **Must** |
| REQ-04 | Ne pas dégrader le parcours de mise en ligne | O | S4 | **Must** |
| REQ-05 | Ne traiter aucune donnée personnelle | R | S2 | **Must** |
| REQ-06 | Tenir sous 200 ms par article | NF | S3 | Should |
| REQ-07 | Basculer en revue humaine sous un seuil de confiance | O | S4 | Should |
| REQ-08 | Tracer la version du modèle sur chaque prédiction | T | S2 | Should |
| REQ-09 | Coût d'inférence maîtrisé et mesuré | NF | S1 | Should |
| REQ-10 | Rester interprétable pour l'équipe métier | O | S4 | Should |
| REQ-11 | Tenir la montée en volume du catalogue | NF | S1 | Should |
| REQ-12 | Absorber l'apparition de nouvelles catégories | T | S1 | Could |
| REQ-13 | Exploiter l'image en complément du texte | F | S1 | Could |
| REQ-14 | Réentraînement automatique en production | T | hypothèse | Won't (v1) |

### 1.5 Hiérarchisation des besoins

Positionnement impact × effort. Quatre quadrants, lecture immédiate.

| | Effort faible | Effort élevé |
|---|---|---|
| **Impact fort** | REQ-01, 02, 03, 05 → **socle v1** | REQ-11, 13 → à planifier |
| **Impact faible** | REQ-08, 09, 10 → gains rapides | REQ-12, 14 → hors périmètre v1 |

Le socle v1 tient en quatre besoins. Tout le reste est explicitement différé, et le motif est écrit.

### 1.6 Couverture des familles d'exigences

| Famille | Nombre | Exemple | Vérification |
|---|---|---|---|
| Fonctionnel | 5 | REQ-01 attribuer une catégorie | F1 macro sur jeu de test |
| Non fonctionnel | 4 | REQ-06 sous 200 ms | Temps d'inférence mesuré de bout en bout |
| Organisationnel | 3 | REQ-07 revue humaine sous seuil | Seuil implémenté dans l'application |
| Technique | 3 | REQ-08 version du modèle tracée | Champ de version en sortie |
| Réglementaire | 1 | REQ-05 aucune donnée personnelle | Revue du corpus, aucune donnée nominative |

Aucune famille n'est vide.

---

## 2. Audit de la solution data

### 2.1 L'objet audité

L'audit porte sur la solution existante : l'étude de faisabilité menée fin 2024, ses données, ses
notebooks et sa chaîne de traitement.

### 2.2 Démarche et référentiel

Trois volets — flux, qualité, architecture. Le référentiel de qualité retenu est le corps de
dimensions **DAMA** (complétude, unicité, validité, cohérence, exactitude, actualité), choisi parce
qu'il est sectoriellement neutre et que chacune de ses dimensions se traduit en test exécutable. Un
référentiel purement documentaire n'aurait pas cette propriété.

Les tests sont automatisés dans `scripts/profile_data.py` et rejouables.

### 2.3 Volet A — Les flux

`description + image` → extraction de caractéristiques → matrices sérialisées → modèle → prédiction

| Saut | Format | Volume | Point de rupture identifié |
|---|---|---|---|
| Source → notebook | CSV + dossier d'images | 1 050 lignes, 1 050 images | Chemin relatif codé en dur hors du dépôt |
| Extraction → stockage | Sérialisation `joblib` | 6 matrices texte, 3 image | **Hors dépôt, non versionné** |
| Stockage → modèle | Chargement direct | 7 153 dimensions | Aucun contrôle de schéma |
| Modèle → prédiction | En mémoire | — | Aucune persistance du modèle entraîné |

### 2.4 Volet B — La qualité des données

Mesures exécutées sur les 1 050 lignes et 15 colonnes du fichier source.

| Dimension | Test | Résultat | Verdict |
|---|---|---|---|
| Complétude | Descriptions manquantes | 0 | ✅ |
| Complétude | Autres champs | `brand` 338 / 1 050 (32 %) ; trois autres champs à 1 manquant | ⚠️ |
| Unicité | Doublons d'identifiant et de description | 0 et 0 | ✅ |
| Validité | Catégories hors nomenclature | 7 catégories, **150 articles chacune** | ✅ |
| Cohérence | Appariement description ↔ image ↔ identifiant | 1 050 / 1 050 | ✅ |
| Exactitude | Catégorie conforme à l'article | **Non vérifiable** : la vérité terrain est déclarée par les vendeurs | ⚠️ |
| Actualité | Fraîcheur du corpus | Extraction 2016 | ⚠️ |

**Distribution de l'information.** Longueur des descriptions en mots : minimum 13, premier quartile
30, médiane 44, troisième quartile 94, maximum 587. La médiane varie fortement selon la catégorie —
24 pour *Home Furnishing*, 88 pour *Kitchen & Dining*. Les classes sont équilibrées en volume, pas en
information.

**Le cas du champ `brand`.** Ses 32 % de valeurs manquantes ne sont pas réparties au hasard : 140 sur
150 pour *Watches*, 109 sur 150 pour *Beauty and Personal Care*, mais **0 sur 150** pour *Computers*
et *Home Furnishing*. L'absence même de la marque prédit la catégorie. Retenir ce champ, ou même un
indicateur de sa présence, ferait fuiter la cible. **Champ exclu du modèle.**

L'exactitude non vérifiable constitue une **borne haute** à toute performance mesurée : le modèle est
évalué contre des étiquettes elles-mêmes bruitées.

### 2.5 Volet C — L'architecture

| Constat | Criticité | Preuve | Action |
|---|---|---|---|
| **AUD-01** — Données non versionnées et **plus disponibles** à la reprise ; les notebooks référencent un dossier absent | **Bloquant** | Aucun artefact intermédiaire retrouvable | Versionner ; figer un échantillon dans le dépôt |
| **AUD-02** — Aucun pipeline : la logique vit dans sept notebooks sans point d'entrée | Majeur | Arborescence du projet initial | Extraire un module appelable, une commande de rejeu |
| **AUD-03** — Le modèle entraîné n'est pas sérialisé | Majeur | Aucun artefact sauvegardé | Sérialiser et versionner |
| **AUD-04** — Comparaison non homogène : un modèle consomme des caractéristiques fusionnées texte + image, les autres des pixels bruts | Majeur | Écarts de 0,46 à 0,97 sur des entrées de nature différente | Protocole unique : même découpe, même tête |
| **AUD-05** — Vérité terrain déclarative, donc bruitée | Mineur | Énoncé du problème | À déclarer ; borne haute de performance |
| **AUD-06** — Aucune trace d'exécution hors notebook | Mineur | — | Journal d'exécution, versions épinglées |

### 2.6 Évaluation de l'adéquation

La solution auditée **démontre la faisabilité** mais **n'est pas exploitable** : elle ne se rejoue
pas, ne produit pas d'artefact réutilisable, et sa comparaison de modèles n'est pas interprétable.
Elle répond à la question « est-ce possible ? » et à aucune autre.

AUD-01 et AUD-02 conditionnent tout le reste : sans reproductibilité, aucune mesure n'est opposable.

---

## 3. Solution technique cible

### 3.1 Identification des cas d'usage

Dérivés du registre des besoins, complétés par les parcours des personas.

| ID | Cas d'usage | Acteur | Déclencheur | Décision produite |
|---|---|---|---|---|
| **UC1** | Suggérer la catégorie à la publication | Vendeur | Saisie de la description | Catégorie pré-remplie, modifiable |
| **UC2** | Recatégoriser le catalogue existant | Responsable data | Traitement par lot | Liste d'articles à corriger |
| **UC3** | Détecter les articles mal catégorisés | Équipe catalogue | Désaccord modèle / déclaration | File de revue |
| **UC4** | Router vers une revue humaine sous seuil | Équipe catalogue | Confiance basse | Article mis en attente |
| **UC5** | Enrichir les facettes de recherche | Acheteur | Requête | Filtres plus fiables |
| **UC6** | Surveiller la dérive du catalogue | Responsable data | Nouveaux articles | Alerte sur catégorie émergente |

### 3.2 Évaluation et hiérarchisation

**Méthode déclarée avant le scoring** : cinq critères, échelle 1 à 5, pondération arrêtée à l'avance,
notation par l'auteur en une passe le jour du cadrage. Score sur 100.

| Critère | Poids | Justification du poids |
|---|---|---|
| Valeur métier | 30 | C'est l'objet du projet |
| Faisabilité avec l'existant | 25 | Contrainte de temps forte |
| Effort de mise en œuvre *(inversé)* | 20 | Différencie les cas proches en valeur |
| Risque en cas d'erreur *(inversé)* | 15 | Une erreur visible coûte plus qu'une erreur interne |
| Effet de démonstration | 10 | Un cas montrable accélère la décision |

| UC | Valeur | Faisabilité | Effort | Risque | Démo | **Total** | Périmètre |
|---|---|---|---|---|---|---|---|
| **UC1** | 5 | 5 | 4 | 4 | 5 | **92** | **v1** |
| **UC4** | 4 | 5 | 4 | 5 | 4 | **87** | **v1** |
| UC3 | 4 | 4 | 3 | 4 | 3 | 74 | v2 |
| UC2 | 4 | 4 | 3 | 3 | 2 | 70 | v2 |
| UC5 | 3 | 3 | 2 | 4 | 2 | 58 | v3 |
| UC6 | 3 | 2 | 2 | 3 | 2 | 50 | v3 |

**Frontière v1 : UC1 et UC4.** Suggérer, et savoir se taire quand la confiance est insuffisante. Les
deux sont indissociables : une suggestion sans garde-fou déplace le problème au lieu de le résoudre.

### 3.3 Comparatif des approches techniques

| Approche | Ce qu'elle apporte | Sa limite | Coût |
|---|---|---|---|
| **Encodeur d'images figé** | Conçu pour l'usage figé, il tient ses promesses : 0,937 seul | Ne voit pas ce que seul le texte dit — marque, dimensions | 346 Mo · 36 ms |
| **Fusion texte + image** | Répare les catégories confondues : classe faible 0,870 → 0,955 | Deux chaînes à maintenir ; dépend de la qualité de la photographie | 365 Mo · 36 ms |
| **TF-IDF** | Interprétable terme à terme ; excellente sur un vocabulaire discriminant | Aucune notion de sens : deux synonymes sont deux dimensions étrangères | Négligeable |
| **Encodeur pré-entraîné figé** | Représentation contextuelle, sans entraînement | La moyenne des tokens d'un modèle masqué est une représentation de phrase médiocre | 438 à 596 Mo |
| **Encodeur réglé finement** | Adapte la représentation au domaine | Demande du temps de calcul et une validation supplémentaire | Non évalué ici |

| Classifieur | Ce qu'il apporte | Sa limite |
|---|---|---|
| **XGBoost** | Robuste sans réglage ; importance des variables lisible | Moins à l'aise en très haute dimension creuse |
| **Perceptron multicouche** | Meilleur compromis mesuré ; probabilités exploitables comme confiance | Boîte noire |

### 3.4 Adéquation de la solution retenue

| Axe | Exigence | Ce que fait la solution | Écart résiduel |
|---|---|---|---|
| Fonctionnalités | REQ-01, 02, 03 | Classification 7 classes, score de confiance | Image non exploitée en v1 |
| Performances | F1 macro ≥ 0,90 ; < 200 ms | 0,943 et 0,06 ms | Aucun |
| Réglementaire | Aucune donnée personnelle | Corpus produit exclusivement | Aucun |
| Sécurité | Pas de donnée sensible | Modèle local, aucune donnée sortante | Authentification hors périmètre |
| Scalabilité | Montée en volume | Coût constant par article, 14 Mo, CPU suffisant | Réentraînement non automatisé |
| Coûts | Maîtrisés et mesurés | Mesurés et comparés entre approches | Hébergement non chiffré |

### 3.5 Architecture cible

```
Fiche produit
     │
     ├─► Vectorisation TF-IDF  (ajustée sur l'entraînement uniquement)
     │
     ├─► Perceptron multicouche  ──►  7 probabilités
     │
     └─► Arbitrage sur le seuil de confiance
              ├─ ≥ seuil  ──►  suggestion au vendeur, modifiable
              └─ < seuil  ──►  file de revue humaine
```

Le point d'arbitrage est la pièce maîtresse : c'est lui qui transforme un classifieur en système
exploitable.

---

## 4. Mise en œuvre et industrialisation

### 4.1 La chaîne de traitement

Cinq étapes, illustrées sur un article réel en `reports/fig3_transformations.png` :
description brute (55 mots) → normalisation (85 jetons) → vocabulaire (4 532 termes, n-grammes 1–2,
seuil de fréquence minimale à 2) → pondération TF-IDF sublinéaire (61 termes non nuls) →
classification (7 probabilités).

Toutes les étapes à état sont ajustées **sur le jeu d'entraînement uniquement**.

### 4.2 Le protocole de validation

Découpe 70 / 15 / 15 stratifiée, graine fixe, définie dans un module unique appelé aussi bien par le
benchmark que par l'application. 735 articles d'entraînement, 157 de validation, 158 de test.

Le seuil d'acceptation métier — **F1 macro ≥ 0,90** — a été arrêté avant toute mesure. C'est une
condition d'interprétabilité : une tolérance dérivée du résultat obtenu place le modèle à la limite
d'acceptation par construction.

Le choix de la F1 macro plutôt que de l'exactitude tient à la lecture par classe : les sept
catégories étant équilibrées, les deux valeurs coïncident presque, mais seule la F1 par classe
localise l'erreur.

### 4.3 Les résultats

| Modèle | F1 macro | Exactitude | F1 classe la plus faible | Entraînement | Inférence | Empreinte | Seuil |
|---|---|---|---|---|---|---|---|
| **Fusion texte + image** | **0,9744** | 0,9747 | **0,955** | 2,0 s | 35,8 ms | 365,4 Mo | ✅ |
| TF-IDF + MLP | 0,9427 | 0,9430 | 0,870 | **0,52 s** | **0,06 ms** | 14,3 Mo | ✅ |
| DINOv2 figé — image seule | 0,9366 | 0,9367 | 0,864 | 0,17 s | 35,7 ms | 346,3 Mo | ✅ |
| BERT figé (2018) | 0,9243 | 0,9241 | 0,870 | 17,3 s | 23,4 ms | 437,9 Mo | ✅ |
| TF-IDF + XGBoost | 0,9234 | 0,9241 | 0,837 | 4,8 s | 0,08 ms | **2,6 Mo** | ✅ |
| ModernBERT figé (2024) | 0,8854 | 0,8861 | 0,800 | 31,6 s | 44,6 ms | 596,1 Mo | ❌ |

**Lecture.** Cinq modèles sur six franchissent le seuil métier. La performance ne départage donc
rien : **le critère de décision devient le coût.**

La fusion gagne 3 points de F1 macro sur le meilleur modèle texte, et 8 points sur la catégorie la
plus faible — de 0,870 à 0,955. Elle coûte 600 fois l'inférence du modèle texte, mais **reste très
en deçà du budget de 200 ms fixé par REQ-06** : la contrainte qui aurait pu l'écarter ne mord pas.
C'est pour cela qu'elle est retenue malgré son coût. Le modèle texte reste le recours quand aucune
photographie n'est disponible, ce que l'application implémente.

Le modèle le plus récent et le plus lourd, ModernBERT figé, est le seul à échouer.

**La limite de cette comparaison, énoncée sans détour.** Les encodeurs sont utilisés **figés**, sans
réglage fin. Ce qui est mesuré est **une représentation, pas une capacité de modèle**. Un réglage fin
changerait probablement le classement, et constitue la première itération suivante.

**Et le verdict diffère selon la modalité, ce qui est un résultat en soi.** DINOv2 est entraîné en
auto-supervision précisément pour produire de bonnes représentations figées : utilisé ainsi, il
atteint 0,937 à lui seul, au-dessus de tous les modèles de texte pré-entraînés. Un modèle de langage
masqué dont on moyenne les jetons n'a jamais été entraîné pour cela, et il déçoit. « Encodeur
pré-entraîné figé » n'est donc pas une catégorie homogène — la question à poser est *pour quel usage
cet encodeur a-t-il été entraîné*.

Côté texte, la conclusion défendable reste étroite : *à protocole constant, sur cette tâche et à ce
volume, une représentation lexicale bat les encodeurs de texte figés sur tous les axes.* La tâche est
lexicalement séparable — le vocabulaire d'une fiche produit est très discriminant — ce qui est
précisément le terrain de la représentation lexicale.

**Par catégorie** (`reports/fig2_f1_par_classe.png`). Les deux classes les plus faibles pour tous les
modèles de texte sont *Home Decor & Festive Needs* (0,80 à 0,87) et *Baby Care* (0,83 à 0,91).
L'étude qualitative de 2024 annonçait ces confusions ; la mesure les confirme.

**C'est exactement là que l'image apporte.** La fusion porte *Home Decor* à 0,955 et *Baby Care* à
0,957. Le résultat est intelligible : un vase et un ustensile de cuisine se décrivent avec un
vocabulaire voisin, mais ne se ressemblent pas en photographie.

**Une convergence à signaler.** L'exactitude de la fusion 2026 s'établit à 0,9747 — la valeur exacte
obtenue en 2024 avec un encodeur d'images entièrement différent. Deux chaînes indépendantes
atteignent le même plafond, ce qui suggère qu'il tient à la qualité des étiquettes plutôt qu'aux
modèles.

### 4.4 Le seuil de confiance

| Seuil | Fusion — couverture | Fusion — erreurs | Texte seul — couverture | Texte seul — erreurs |
|---|---|---|---|---|
| 0,50 | 92,4 % | 1,4 % | 91,1 % | 2,8 % |
| **0,60** | **82,9 %** | **aucune sur 131 articles** | 84,2 % | 2,3 % |
| 0,70 | 71,5 % | aucune sur 113 | 76,6 % | 0,8 % |
| 0,80 | 62,0 % | aucune sur 98 | 67,7 % | aucune sur 107 |

**Seuil recommandé : 0,60 sur la fusion.** Elle atteint zéro erreur observée dès ce seuil, en
couvrant 83 % du volume ; le modèle texte doit monter à 0,80 pour en faire autant et ne couvre alors
que 68 %. L'image achète donc 15 points de couverture à qualité égale.

La formulation prudente est exigée : « aucune erreur sur les 131 articles concernés », et non
« 100 % de précision » — sur cet effectif, une borne de confiance est plus honnête qu'un chiffre
rond.

### 4.5 Risques et opportunités

| Risque | Criticité | Mesure d'atténuation |
|---|---|---|
| Vérité terrain bruitée (étiquettes déclarées) | Majeur | Déclaré comme borne haute ; échantillon à ré-étiqueter pour lever le doute |
| Modèle appris sur un corpus de 2016 | Majeur | Réentraînement sur données récentes avant tout déploiement |
| Sur-confiance de l'opérateur | Moyen | Confiance affichée, revue obligatoire sous seuil |
| Catégories émergentes non couvertes | Moyen | Surveillance du taux d'abstention comme signal |
| Dépendance à la longueur de description | Mineur | Documenté ; les descriptions courtes concentrent l'incertitude |

**Opportunités.** La fusion texte + image reste la meilleure performance connue sur ce jeu et n'a pas
été réexploitée ici. Les corrections apportées par les vendeurs constituent un signal de
réentraînement gratuit et continu.

### 4.6 Coûts

| Poste | Estimation |
|---|---|
| Étude de faisabilité initiale | ~5 jours |
| Reprise, benchmark, industrialisation, rapport | 1 jour |
| Infrastructure de développement | 0 € — exécution locale |
| Inférence en production | Modèle de 14 Mo, CPU suffisant, 0,06 ms par article |
| **Facteur de coût dominant** | Le choix de la représentation. Un encodeur profond impose un GPU ou un service dédié ; la solution retenue tient sur un CPU partagé |

### 4.7 Impacts

| Famille | Impact | Sens | Maîtrise | Indicateur de suivi |
|---|---|---|---|---|
| **Réglementaire** | Aucune donnée personnelle dans le corpus | + | Revue documentée à chaque nouveau corpus | Contrôle avant ingestion |
| **Éthique** | Biais de catégorisation : les catégories mal servies le restent | − | F1 **macro** et suivi par classe, jamais l'exactitude seule | F1 de la classe la plus faible |
| **Business** | Une erreur rend l'article invisible : perte de vente directe | − | Seuil de confiance et revue humaine | Erreurs sur la part automatisée |
| **Organisationnel** | Le geste du vendeur évolue : il valide au lieu de choisir | ± | Suggestion modifiable, jamais imposée | Taux de correction manuelle |
| **Humain** | Sur-confiance : l'opérateur cesse de vérifier | − | Confiance affichée, abstention sous seuil | Part des prédictions revues |

---

## 5. Contrôle et suivi du projet

### 5.1 Méthodologie

Kanban en solo, cadence journalière. Scrum n'a pas été retenu : ni équipe, ni vélocité mesurable sur
ce format. *Definition of Done* en trois points : le code se rejoue de bout en bout, le chiffre est
reporté dans le rapport, la limite est écrite.

### 5.2 Jalons — prévu contre réel

| Jalon | Critère de fin | Prévu | Réel | Écart | Cause |
|---|---|---|---|---|---|
| **J1** Faisabilité | Accord clusters / catégories mesuré | nov. 2024 | déc. 2024 | +2 sem. | Volume de représentations exploré supérieur au prévu |
| **J2** Premier supervisé | Trois modèles chiffrés | déc. 2024 | déc. 2024 | 0 | — |
| **J3** Reprise | Données récupérées, environnement reconstruit | 1 h | 3 h | **+2 h** | **Données non versionnées — AUD-01** |
| **J4** Benchmark | Six modèles comparés à protocole constant, texte et image | 3 h | 4 h | **+1 h** | Extraction des caractéristiques visuelles ajoutée en cours de route |
| **J5** Démonstration | Application fonctionnelle, scénarios vérifiés | 1 h | 1 h | 0 | — |
| **J6** Livrables | Dépôt public, rapport, figures | 2 h | 2 h | 0 | — |

L'écart de J3 est le plus instructif du tableau : il chiffre le coût d'un défaut d'architecture
identifié à l'audit, et il justifie à lui seul la recommandation R1.

### 5.3 Indicateurs de pilotage

| KPI | Définition | Cible | Mesuré | Verdict |
|---|---|---|---|---|
| **P1** F1 macro | Moyenne non pondérée des F1 par classe | ≥ 0,90 | **0,9744** | ✅ |
| **P2** F1 de la classe la plus faible | Minimum par classe | ≥ 0,80 | **0,955** | ✅ |
| **P3** Temps d'inférence | ms par article, de bout en bout | < 200 ms | **35,8 ms** | ✅ |
| **P4** Temps d'entraînement | Secondes, du brut au modèle | — | **2,0 s** | mesuré |
| **P5** Empreinte déployée | Mo du modèle et de son encodeur | — | **365,4 Mo** | mesuré |
| **P6** Couverture au seuil 0,60 | Part d'articles classés automatiquement | ≥ 60 % | **82,9 %** | ✅ |
| **P7** Erreurs sur la part automatisée | Taux d'erreur au-dessus du seuil | ≤ 1 % | **0 / 131** | ✅ |
| **I1** Reproductibilité | Le projet se rejoue depuis le dépôt | oui | **oui** (`make all`) | ✅ |
| **I2** Traçabilité | Version du modèle sur chaque prédiction | oui | **oui** — empreinte SHA-256 tronquée | ✅ |

Tous les indicateurs sont au vert. I2 a été traité en cours de projet : chaque prédiction porte désormais l'empreinte du modèle qui l'a produite.

### 5.4 Journal des itérations

| It | Hypothèse testée | Changement | Avant | Après | Décision |
|---|---|---|---|---|---|
| It-0 | Le texte seul sépare les catégories | Représentation lexicale seule | — | accord 0,37 | référence retenue |
| It-1 | L'image apporte de l'information | Ajout des caractéristiques visuelles profondes | 0,37 | 0,35 seule | conservée pour la fusion |
| It-2 | Les descripteurs locaux apportent de l'information | Ajout des points d'intérêt | 0,42 | **0,38** | **annulé — dégrade** |
| It-3 | La fusion bat chaque modalité seule | Texte + image, sans descripteurs locaux | 0,37 | **0,42** | conservé |
| It-4 | Le supervisé dépasse le non supervisé | Perceptron sur caractéristiques fusionnées | 0,42 | **97 % d'exactitude** | conservé |
| It-5 | L'augmentation de données aide les modèles image | Rotations, zooms, décalages | 0,854 | **0,722** | **annulé — dégrade** |
| It-6 | La réduction de dimension préserve la performance | 7 153 → 432 dimensions | — | 95 % de variance | conservé |
| It-7 | Un encodeur pré-entraîné bat le lexical | BERT figé + tête linéaire | 0,943 | **0,924** | **annulé — plus coûteux et moins bon** |
| It-8 | Un encodeur récent bat BERT | ModernBERT figé | 0,924 | **0,885** | **annulé — sous le seuil** |
| It-9 | Un seuil de confiance permet d'automatiser sans erreur | Abstention sous 0,80 | 9 erreurs / 158 | **0 / 107, 68 % couverts** | conservé |
| It-10 | Un encodeur figé est meilleur sur l'image que sur le texte | DINOv2 figé, image seule | 0,924 (BERT figé) | **0,937** | conservé — le verdict dépend de la modalité |
| It-11 | La fusion répare les catégories que le texte confond | Concaténation après normalisation L2 | 0,870 sur la classe faible | **0,955** | **conservé — modèle retenu** |

Quatre itérations sur douze ont été **annulées**. C'est la partie du journal qui a le plus de
valeur : elle montre ce qui a été tenté puis abandonné, pas seulement ce qui a fonctionné.

---

## 6. Conclusion et recommandations

### 6.1 Ce qui est démontré, ce qui ne l'est pas

| Démontré | Non démontré |
|---|---|
| La catégorisation automatique atteint le seuil métier fixé d'avance | Que ce seuil soit le bon : il relève d'un arbitrage produit |
| Le coût varie de trois ordres de grandeur à performance comparable | Qu'un encodeur réglé finement ne renverse pas le classement |
| L'image répare les catégories que le texte confond : 0,870 → 0,955 | Que le gain tienne sur un catalogue où les photographies sont moins soignées |
| Un seuil à 0,60 sur la fusion automatise 83 % du volume sans erreur observée | Que ce taux tienne hors du jeu de test (131 articles) |
| La chaîne se rejoue intégralement en une commande | Que le modèle tienne sur un catalogue actuel : le corpus date de 2016 |
| Les catégories confuses sont identifiées et constantes | Que leur confusion soit réductible sans données supplémentaires |

### 6.2 Recommandations

| # | Recommandation | Destinataire | Effort | Preuve attendue |
|---|---|---|---|---|
| **R1** | Versionner les données et figer un échantillon dans le dépôt avant toute reprise | Responsable data | 1 j | Rejeu complet depuis un clone vierge |
| **R2** | Déployer UC1 en suggestion modifiable, jamais en attribution silencieuse | Produit | 3 j | Taux d'acceptation de la suggestion |
| **R3** | Fixer le seuil à 0,80 et mesurer la couverture réelle en production | Responsable data | 1 j | Courbe couverture / erreurs sur données récentes |
| **R4** | Retenir la fusion quand une photographie existe, le modèle texte sinon — le budget de latence l'autorise | Responsable data | 1 j | Tableau §4.3 et §4.4 |
| **R5** | Étendre la traçabilité au journal des prédictions, pas seulement à l'affichage | Ingénierie | 0,5 j | Journal horodaté, empreinte du modèle par ligne |
| **R6** | Régler finement un encodeur avant de conclure sur les transformeurs de texte | Ingénierie | 2 j | Comparaison à protocole égal |
| **R7** | Vérifier que le gain de l'image tient sur des photographies de qualité hétérogène | Responsable data | 2 j | F1 par classe sur un lot récent |
| **R8** | Prévoir le repli texte quand la photographie manque ou est illisible | Produit | 0,5 j | Taux d'articles sans photographie exploitable |

### 6.3 Prochaines étapes

**0 à 1 mois** — R1 et R5, puis déploiement de UC1 et UC4 en suggestion assistée.
**1 à 3 mois** — R6, R7 et R8 ; collecte des corrections vendeurs comme signal de réentraînement.
**Au-delà** — surveillance de la dérive du catalogue (UC6) et réentraînement gradué.

---

## 7. Annexes

### A. Reproductibilité

```bash
pip install -r requirements.txt
make all                     # benchmark + figures, une dizaine de secondes
make benchmark ENCODERS=1    # ajoute les encodeurs pré-entraînés
make demo                    # application de démonstration
```

Versions épinglées dans `requirements.txt`. Graine fixe, découpe centralisée dans `src/pipeline.py`.

### B. Figures

| Figure | Contenu |
|---|---|
| `reports/fig1_cout.png` | Performance contre coût d'inférence, échelle logarithmique |
| `reports/fig2_f1_par_classe.png` | F1 par catégorie et par modèle — l'image répare les classes faibles |
| `reports/fig3_transformations.png` | La chaîne de transformation sur un article réel |
| `reports/fig4_donnees.png` | Équilibre des classes et distribution des longueurs |

### C. Données produites

| Fichier | Contenu |
|---|---|
| `reports/benchmark.csv` | Le tableau de résultats complet |
| ` reports/f1_par_classe.json` | F1 détaillée par classe et par modèle |

### D. Note sur les métriques

Aucune métrique n'est bonne dans l'absolu ; chacune répond à une question précise.

| Métrique | Sa question | Quand elle induit en erreur |
|---|---|---|
| Exactitude | Quelle part de prédictions justes ? | Dès que les classes sont déséquilibrées — ce n'est pas le cas ici |
| F1 macro | Toutes catégories traitées à égalité ? | Masque quelle classe décroche : c'est une moyenne |
| F1 par classe | Où se concentre l'erreur ? | Bruitée sur petits effectifs — 22 articles par classe au test |
| Temps d'inférence | Le passage à l'échelle est-il tenable ? | Ment si la vectorisation ou l'encodage sont exclus du chronomètre |
| Empreinte | CPU ou GPU ? | Ignore la mémoire vive réellement consommée |
| Couverture au seuil | Quel volume est automatisable ? | Sans son taux d'erreur associé, elle ne veut rien dire |
