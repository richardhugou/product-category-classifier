# Une IA de triage pour le catalogue Flipkart

**Étude de faisabilité — classification automatique d'articles à partir du texte et de l'image**

Richard Hugou · août 2026

---

## Ce que vous lirez ici

Flipkart laisse ses vendeurs ranger eux-mêmes leurs articles dans le catalogue. Ça marche mal, et ça marchera de plus en plus mal à mesure que le catalogue grossit. La question posée est simple : **une machine peut-elle faire ce rangement à leur place ?**

La réponse courte est oui, à trois conditions qui font tout l'intérêt de l'étude. Il faut lire la photographie autant que la description. Il faut accepter que le système se taise quand il n'est pas sûr. Et il faut savoir que le meilleur modèle du marché n'est pas forcément le bon : sur cette tâche, le plus récent et le plus lourd est aussi le seul à échouer.

Le chiffre à retenir : **83 % du catalogue peut être rangé automatiquement sans une seule erreur observée**, le reste partant en révision humaine.

---

# 1. Le problème métier

## 1.1 Comment fonctionne une place de marché

Flipkart est l'une des grandes plateformes de commerce en ligne indiennes. Son modèle n'est pas celui d'un magasin : Flipkart ne possède pas la plupart des articles qu'elle affiche. Elle héberge des dizaines de milliers de vendeurs indépendants — un artisan de Jaipur, un importateur de montres, une boutique de puériculture — qui publient eux-mêmes leurs fiches produit.

Chaque fiche comporte trois choses : un nom, une description libre, une photographie. Et une quatrième, qui est celle qui nous intéresse : **une catégorie, choisie par le vendeur dans l'arborescence du site**.

Cette arborescence est profonde. Un drap de lit se range dans `Home Furnishing >> Bed Linen >> Bedsheets`. Le vendeur doit descendre l'arbre et choisir. C'est un geste manuel, effectué par quelqu'un dont ce n'est pas le métier, sur une nomenclature qu'il ne connaît pas bien, et qu'il refait à chaque article.

## 1.2 Ce qui casse

Un vendeur pressé se trompe de branche. Un vendeur malin se range volontairement dans une catégorie plus fréquentée. Un vendeur étranger ne comprend pas la nomenclature. Le résultat est le même dans les trois cas : **l'article est au mauvais endroit du catalogue**.

Et un article mal rangé est un article invisible. L'acheteur qui filtre par catégorie ne le voit pas. Le moteur de recherche interne, qui s'appuie sur ces catégories pour ordonner ses résultats, le classe mal. Personne ne le trouve, personne ne l'achète.

Le coût se répartit sur trois épaules :

Le **vendeur** perd une vente sans jamais savoir pourquoi. Il ne reçoit aucun signal : son article est en ligne, il est simplement introuvable. C'est la pire forme d'échec, celle qu'on ne peut pas corriger parce qu'on ne la voit pas.

L'**acheteur** tombe sur des rayons incohérents. Une catégorie « Kitchen & Dining » qui contient un jouet érode la confiance dans tout le classement, et l'acheteur cesse d'utiliser les filtres.

**Flipkart** perd du chiffre d'affaires sur des articles qu'elle héberge déjà, et paye des équipes catalogue pour repasser derrière.

## 1.3 Pourquoi maintenant

Le problème existe depuis toujours mais restait tenable tant que le catalogue était petit. Il ne l'est plus : l'ouverture à de nouveaux vendeurs fait croître le volume plus vite que la capacité de contrôle manuel. Le contrôle humain a un coût linéaire — deux fois plus d'articles, deux fois plus d'heures. Un modèle a un coût quasi nul à l'article une fois qu'il existe.

C'est cette divergence de courbes qui déclenche la demande, pas une insatisfaction nouvelle.

## 1.4 Ce qu'on nous demande

La demande est formulée comme une étude de faisabilité, et il faut la prendre au mot. On ne nous demande pas de déployer un système. On nous demande de répondre à quatre questions :

1. Est-il possible d'attribuer automatiquement la catégorie d'un article à partir de ce qu'on a — sa description et sa photographie ?
2. Avec quelle fiabilité ?
3. À quel coût, et ce coût tient-il si le catalogue est cent fois plus gros ?
4. Dans quelles conditions peut-on se passer d'un humain, et dans quelles conditions ne le peut-on pas ?

La quatrième question est la plus importante et c'est celle qu'on oublie le plus souvent. Un modèle à 97 % de bonnes réponses n'est pas un modèle utilisable tant qu'on ne sait pas **dans quels cas** il se trompe, ni comment le savoir au moment où il répond.

---

# 2. Qui écrit ce rapport

Autant le dire d'emblée, parce que ça détermine ce qui suit : **je suis data scientist junior**, et ce travail est un travail de junior.

Concrètement, cela veut dire trois choses.

**Ce que je peux faire.** Construire une chaîne de traitement complète et reproductible, comparer honnêtement plusieurs approches à protocole constant, mesurer ce qui compte, et livrer une recommandation argumentée avec ses conditions de validité.

**Ce que je ne prétends pas faire.** Concevoir une architecture qui tient à l'échelle du catalogue réel de Flipkart. Négocier les arbitrages produit — le seuil de confiance acceptable est une décision métier, pas une décision technique. Garantir qu'un modèle appris sur mille articles se comportera pareil sur un million.

**Ce sur quoi j'appelle une relecture senior.** Le passage à l'échelle, la robustesse en production, et la question du réglage fin des grands modèles, qui demande des moyens de calcul et une expertise que cette étude n'a pas mobilisés.

Un mot sur le cadre : l'étude est menée sur **un échantillon public de 1 050 articles Flipkart**, dans un cadre de mise en situation. Les conclusions sont valables sur cet échantillon et le rapport dit à chaque fois ce qui se transpose et ce qui ne se transpose pas. Un junior qui conclurait « le système est prêt » sur mille articles se tromperait de métier.

---

# 3. La solution, vue de loin

Avant d'entrer dans la technique, voici ce qu'on veut construire, décrit comme on le décrirait à un vendeur.

**Une IA de triage.** Le vendeur remplit sa fiche : il tape sa description, il téléverse sa photo. Au moment où il arrive au champ « catégorie », le champ est déjà rempli. Il n'a plus qu'à vérifier — et corriger si c'est faux.

Trois principes gouvernent ce choix, et ils comptent plus que le modèle qui sera dessous.

**On suggère, on n'impose pas.** La catégorie proposée reste modifiable. Le vendeur garde la main. C'est ce qui rend le système acceptable, et c'est aussi ce qui le rend améliorable : chaque correction d'un vendeur est une donnée d'apprentissage gratuite.

**On se tait quand on ne sait pas.** Le modèle ne renvoie pas seulement une catégorie, il renvoie aussi sa confiance. En dessous d'un certain niveau, il n'affiche rien et l'article part dans une file de révision humaine. C'est contre-intuitif — on a construit un système pour qu'il réponde — mais c'est exactement ce qui permet de garantir un taux d'erreur sur la partie automatisée. Un système qui répond toujours ne peut rien garantir.

**On lit ce que le vendeur a produit, rien d'autre.** Pas de données de navigation, pas de profil vendeur, pas d'historique d'achat. La description et la photographie suffisent, et cette frugalité est un choix : elle rend le système explicable, elle évite toute donnée personnelle, et elle fait que le modèle marche dès le premier article d'un vendeur inconnu.

Reste à savoir comment on fabrique une telle chose. Et ça commence par les données.

---

# 4. Le gisement de données

## 4.1 Ce dont on dispose

Un fichier tabulaire de **1 050 articles** et 15 colonnes, plus **1 050 photographies**, une par article. C'est peu. C'est même très peu au regard des standards de l'apprentissage profond, et ce constat conditionnera toutes les décisions techniques qui suivent.

Voici un article réel, tel qu'il arrive :

| Champ | Contenu |
|---|---|
| `product_name` | SANTOSH ROYAL FASHION Cotton Printed King sized Double Bedsheet |
| `product_category_tree` | `["Home Furnishing >> Bed Linen >> Bedsheets >> SANTOSH ROYAL FASHION Bedsheets >> ..."]` |
| `description` | *Key Features of SANTOSH ROYAL FASHION Cotton Printed King sized Double Bedsheet Royal Bedsheet Perfact for Wedding & Gifting, Specifications of ... General Brand SANTOSH ROYAL FASHION Machine Washable Yes Type Flat Material Cotton ...* |
| `image` | une photographie du drap, sur fond blanc |
| `brand` | SANTOSH ROYAL FASHION |
| `retail_price`, `discounted_price`, `product_rating`… | onze autres champs |

## 4.2 De quoi est faite la description

C'est le champ central, et il n'a rien d'un texte rédigé. C'est un **agrégat semi-structuré** : le vendeur a rempli un formulaire de spécifications, et la plateforme a aplati ce formulaire en une chaîne de caractères. On y trouve le nom répété plusieurs fois, une liste de caractéristiques, des paires attribut-valeur collées les unes aux autres, et parfois des fautes de frappe (« Perfact »).

Sa longueur varie énormément :

| | Mots |
|---|---|
| minimum | 13 |
| 1ᵉʳ quartile | 30 |
| **médiane** | **44** |
| 3ᵉ quartile | 94 |
| maximum | 587 |

Un facteur 45 entre l'article le plus bavard et le plus laconique. Et cette variation n'est pas aléatoire : elle dépend de la catégorie. Un drap se décrit en 24 mots médians, un ustensile de cuisine en 88.

| Catégorie | Longueur médiane |
|---|---|
| Home Furnishing | 24 mots |
| Beauty and Personal Care | 34 |
| Computers | 37 |
| Home Decor & Festive Needs | 45 |
| Watches | 46 |
| Baby Care | 57 |
| Kitchen & Dining | 88 |

**Conséquence directe :** les catégories sont équilibrées en nombre d'articles, mais pas en quantité d'information. Certaines classes donnent au modèle trois fois moins de matière que d'autres. C'est une des raisons pour lesquelles la performance ne sera pas uniforme d'une catégorie à l'autre, et pourquoi on mesurera la performance **par catégorie** et pas seulement en moyenne.

## 4.3 Ce qu'on doit prédire

La cible se cache dans `product_category_tree`, qui est une chaîne mal formée contenant un chemin d'arborescence. On n'en garde que **le premier niveau**, ce qui donne sept catégories :

Baby Care · Beauty and Personal Care · Computers · Home Decor & Festive Needs · Home Furnishing · Kitchen & Dining · Watches

**150 articles par catégorie, exactement.** Le jeu est parfaitement équilibré — c'est un échantillon construit, pas un extrait naturel du catalogue. C'est une facilité qu'il faut avoir en tête : sur le vrai catalogue, certaines catégories seraient dix fois plus peuplées que d'autres, et une partie des conclusions demanderait à être revérifiée.

Pourquoi seulement le premier niveau ? Parce qu'avec 1 050 articles, descendre d'un cran donnerait des dizaines de classes à quelques articles chacune. On n'apprend rien de fiable sur cinq exemples. Sept classes à 150 exemples, c'est déjà juste ; soixante-dix classes à quinze exemples ne mènerait nulle part. C'est la première décision imposée par la taille du gisement.

## 4.4 Ce qu'on doit prédire est-il fiable ?

Non, et c'est la limite la plus profonde de toute l'étude.

La catégorie qui sert de vérité de référence est… **celle déclarée par le vendeur**. Or c'est précisément parce que les vendeurs se trompent qu'on construit ce système. On entraîne donc un modèle à reproduire un étiquetage dont on sait qu'il est imparfait.

Cela ne rend pas l'exercice vain — la majorité des vendeurs range correctement, et le signal domine largement le bruit — mais cela pose **un plafond**. Quand le modèle sera en désaccord avec l'étiquette, on ne saura pas lequel des deux a raison. Toute performance mesurée ici est mesurée contre une référence bruitée, et le chiffre affiché est donc probablement une sous-estimation légère de la vraie qualité.

Nous verrons en §8.4 un indice troublant qui suggère que ce plafond a été atteint.

---

# 5. Que fait-on de ces données ? (EDA et ETL)

## 5.1 On ne les prend pas brutes. Voici pourquoi.

La tentation, quand on débute, est de jeter le fichier dans un modèle et de regarder ce qui sort. L'exploration montre pourquoi ça ne marcherait pas — et surtout, elle montre un piège qui aurait produit un excellent score parfaitement faux.

Quatre décisions sortent de cette phase.

## 5.2 Décision 1 — extraire la cible de l'arborescence

`product_category_tree` arrive sous cette forme :

```
["Home Furnishing >> Bed Linen >> Bedsheets >> SANTOSH ROYAL FASHION Bedsheets >> ..."]
```

Ce n'est ni du JSON valide, ni un champ propre : c'est une liste sérialisée à la main, avec des guillemets échappés de façon incohérente. On retire l'enveloppe, on coupe au premier `>>`, on nettoie les espaces. Sept valeurs distinctes en sortent, 150 articles chacune — l'absence de valeur aberrante confirme que le parsing est correct.

Cette opération vit dans **une seule fonction**, appelée par tout le reste du projet. C'est un point d'architecture qui paraît anodin et qui ne l'est pas : si le benchmark et l'application découpaient les données chacun de leur côté, leurs chiffres ne seraient pas comparables et personne ne s'en apercevrait.

## 5.3 Décision 2 — écarter le champ `brand`, qui est un piège

Le champ `brand` est renseigné pour 712 articles sur 1 050. Il manque donc dans 32 % des cas. Le réflexe serait de le garder quand même et d'ajouter un indicateur « marque absente », ce qui est une bonne pratique habituelle.

Ici, ce serait une faute. Regardons **où** il manque :

| Catégorie | Marque absente |
|---|---|
| Watches | **140 / 150** |
| Beauty and Personal Care | **109 / 150** |
| Kitchen & Dining | 71 / 150 |
| Baby Care | 16 / 150 |
| Home Decor & Festive Needs | 2 / 150 |
| Computers | **0 / 150** |
| Home Furnishing | **0 / 150** |

L'absence n'est pas aléatoire, elle est structurée. Parmi les 338 articles sans marque, **95 % appartiennent à seulement trois catégories** — montres, beauté, cuisine. Et symétriquement, un article de *Computers* ou de *Home Furnishing* a toujours sa marque renseignée, sans une seule exception sur 300 articles. **Le fait même que la case soit vide prédit la catégorie.**

Un modèle qui reçoit cette information apprendra à lire le vide plutôt que le produit. Il obtiendrait un score flatteur sur ce jeu de données, et s'effondrerait le jour où un vendeur de montres se met à renseigner sa marque. C'est le cas d'école de la **fuite de données** : une variable qui contient de l'information sur la cible pour une raison qui n'a rien à voir avec le phénomène qu'on veut modéliser.

Le champ est écarté, indicateur d'absence compris.

## 5.4 Décision 3 — normaliser le texte sans le mutiler

Le texte est passé en minuscules, les mots-outils anglais retirés (*the*, *of*, *and*…), et découpé en unigrammes et bigrammes. Les bigrammes comptent : *baby* et *care* pris séparément sont ambigus, *baby care* ne l'est pas ; *hair* et *dryer* isolés se retrouvent dans plusieurs rayons, *hair dryer* dans un seul.

On ne fait **pas** de racinisation ni de lemmatisation. Sur un vocabulaire de fiches produit, `bedsheet` et `bedsheets` portent la même information et le modèle apprend les deux sans difficulté ; la racinisation apporterait peu et détruirait des références de modèles.

On ne corrige **pas** les fautes de frappe. Elles sont rares, et une faute récurrente chez un vendeur est une information comme une autre.

## 5.5 Décision 4 — préparer les images pour un encodeur

Les photographies sont hétérogènes en taille et en format. Chacune est convertie en RGB, redimensionnée et normalisée selon les statistiques attendues par l'encodeur qui les lira — c'est-à-dire qu'on lui présente les images exactement comme celles sur lesquelles il a été entraîné. Sans cette étape, l'encodeur travaille hors de son domaine et ses représentations perdent leur sens.

## 5.6 La découpe des données, faite une fois pour toutes

Avant tout modèle, on coupe le jeu en trois, une fois, et on n'y revient plus :

| | Articles | Rôle |
|---|---|---|
| **Entraînement** | 735 (70 %) | le modèle apprend là-dessus |
| **Validation** | 157 (15 %) | on y choisit les réglages (§9) |
| **Test** | 158 (15 %) | on ne l'ouvre qu'à la fin, une seule fois |

La découpe est **stratifiée** — chaque catégorie garde sa proportion dans les trois parts — et **à graine fixe**, donc identique à chaque exécution. Un test automatisé vérifie qu'aucun article ne se retrouve dans deux parts à la fois.

Cette séparation en trois est la garantie centrale du rapport. Le jeu de test n'est pas un jeu d'évaluation qu'on consulte en cours de route : c'est un jeu qu'on ouvre une fois, avec un seul modèle, pour produire le chiffre qu'on publiera. Chaque coup d'œil supplémentaire l'use, parce qu'on finit par choisir ce qui marche dessus.

---

# 6. Ce que les données racontent

Quatre figures, chacune répondant à une question qu'on se pose à ce stade. Elles sont produites par le code du dépôt et se régénèrent en une commande.

## 6.1 « Le jeu est-il équilibré ? » — oui, et c'est un piège

`fig4_donnees.png` juxtapose deux choses qui ne racontent pas la même histoire. À gauche, sept barres strictement égales à 150 : l'équilibre parfait. À droite, des boîtes à moustaches très inégales : la longueur des descriptions varie du simple au quadruple selon la catégorie.

**On peut donc être équilibré en effectif et profondément déséquilibré en information.** C'est la première chose qu'un tableau de comptage ne dit pas, et c'est ce qui explique pourquoi la performance ne sera pas uniforme entre catégories. C'est aussi ce qui justifie de mesurer par catégorie et pas seulement en moyenne (§8.1).

## 6.2 « Que devient concrètement un article ? »

`fig3_transformations.png` suit un article unique le long de toute la chaîne : 55 mots de description brute → 85 jetons après découpage → un vocabulaire de plusieurs milliers de termes dont **61 seulement sont non nuls pour cet article** → 7 probabilités en sortie.

C'est la figure à montrer à une équipe métier. « Vectoriser » est un mot qui perd tout le monde ; voir qu'un texte devient une liste de 61 poids sur des mots identifiables le rend immédiatement concret — et rend visible ce que le modèle a réellement sous les yeux.

## 6.3 « Où sont les erreurs ? » — pas là où on croit

`fig2_f1_par_classe.png` est la figure la plus utile du rapport, parce qu'elle contredit la lecture moyenne.

Deux catégories décrochent systématiquement, quel que soit le modèle texte employé : **Home Decor & Festive Needs** et **Baby Care**. Ce n'est pas du bruit — c'est constant sur les six approches testées. Ces deux-là partagent un trait : ce sont des catégories définies par l'usage plutôt que par l'objet. Un bougeoir n'est pas un type d'objet, c'est un objet dans un contexte. Le vocabulaire ne le distingue pas d'un objet de cuisine.

Et la même figure montre la solution : la fusion avec l'image ramène ces deux catégories au niveau des autres. **Le problème et son remède sont sur le même graphique.**

## 6.4 « Combien ça coûte ? » — l'axe qui décide

`fig1_cout.png` place chaque modèle dans un plan performance × coût d'inférence, en échelle logarithmique — obligatoire, puisque les coûts s'étalent sur trois ordres de grandeur.

C'est le graphique qui porte la recommandation. Il rend visible un fait que le tableau de scores masque : les modèles s'alignent presque tous sur la même bande horizontale de performance, mais s'étalent sur toute la largeur en coût. Quand tout le monde réussit, **le coût devient le seul critère qui discrimine encore**.

---

# 7. Les modèles : comment ça marche

C'est ici qu'on entre dans la mécanique. La question à laquelle cette section répond est toujours la même, posée pour chaque brique : **qu'est-ce qui rentre, qu'est-ce qui sort ?**

## 7.1 Le principe commun : représenter, puis décider

Aucun modèle ne lit du texte ni ne regarde une image. Tous manipulent des nombres. Toute la chaîne se ramène donc à deux étages :

```
  Article                                                    Décision
     │                                                          ▲
     │   ÉTAGE 1 — REPRÉSENTER          ÉTAGE 2 — DÉCIDER       │
     │   transformer en vecteur         apprendre la frontière  │
     ▼                                                          │
  description ──────► [ 0.0  0.31  0.0  …  0.12 ] ──────► 7 probabilités
  photographie        un vecteur de nombres              qui somment à 1
```

**Étage 1, la représentation.** Elle transforme un article en une liste de nombres. C'est là que se joue l'essentiel, et c'est là qu'on a le plus de choix.

**Étage 2, la tête de classification.** Elle apprend, à partir des 735 articles d'entraînement, à tracer des frontières dans cet espace de nombres. En sortie, sept probabilités qui somment à 1 — une par catégorie.

Séparer ces deux étages est ce qui rend la comparaison honnête : **on garde la même tête et on ne fait varier que la représentation**, ou l'inverse. Sinon on ne sait jamais à quoi attribuer une différence de score.

## 7.2 Les représentations testées

### TF-IDF — compter les mots, intelligemment

**Entrée :** la description, une chaîne de caractères.
**Sortie :** un vecteur de **4 532 nombres**, dont environ 60 seulement sont non nuls.

Le principe tient en deux idées. La première, *term frequency* : plus un mot apparaît dans une description, plus il pèse. La seconde, *inverse document frequency* : plus un mot est répandu dans tout le catalogue, moins il vaut. `product` apparaît partout et ne dit rien ; `bedsheet` apparaît rarement et dit tout. Le poids d'un terme est le produit des deux.

Chaque dimension du vecteur **est** un terme du vocabulaire. C'est la propriété décisive de cette représentation : on peut ouvrir le modèle et lire quels mots ont déclenché quelle décision. Aucune des représentations suivantes ne le permet.

Sa faiblesse est le revers exact de sa force : elle ne connaît aucun sens. *Sofa* et *couch* sont deux dimensions étrangères l'une à l'autre. Un article qui utilise un synonyme absent du vocabulaire d'entraînement est invisible.

### BERT et ModernBERT figés — représenter le sens

**Entrée :** la description, tronquée à 256 jetons.
**Sortie :** un vecteur de **768 nombres**, tous non nuls.

Ces modèles ont été pré-entraînés sur d'immenses corpus à deviner des mots masqués. Ils produisent, pour chaque jeton, une représentation qui dépend du contexte : le mot *mouse* n'a pas le même vecteur dans une fiche informatique et dans une fiche animalerie. On moyenne ensuite ces représentations pour obtenir un vecteur unique par article.

**Ils sont utilisés figés** — leurs poids ne sont jamais modifiés. On les traite comme des extracteurs de caractéristiques, pas comme des modèles à entraîner. Ce choix est dicté par la taille du gisement : régler finement un modèle de 110 millions de paramètres sur 735 exemples est le chemin le plus court vers le surapprentissage, et demande des moyens de calcul dont cette étude ne dispose pas.

C'est la limite majeure de la comparaison et elle est assumée : **on compare des représentations, pas des capacités de modèles.** Cette phrase doit être lue chaque fois qu'un résultat de transformeur apparaît.

Deux générations sont testées, BERT (2018) et ModernBERT (2024), pour voir si sept ans de progrès se traduisent dans cet usage figé.

### DINOv2 figé — représenter une image

**Entrée :** la photographie, redimensionnée en 224 × 224 pixels sur trois canaux.
**Sortie :** un vecteur de **1 536 nombres**.

DINOv2 découpe l'image en carrés de 14 pixels de côté, produit une représentation pour chacun, plus une représentation globale. On conserve la représentation globale **concaténée à** la moyenne des représentations locales — soit 768 + 768. C'est le protocole d'évaluation que ses auteurs recommandent quand le modèle est utilisé figé.

Le point important : contrairement aux modèles de texte ci-dessus, DINOv2 a été **entraîné exprès pour l'usage figé**. Son objectif d'apprentissage auto-supervisé vise précisément à produire des représentations réutilisables telles quelles. On verra en §8.3 que ça change tout.

### La fusion — lire les deux à la fois

**Entrée :** le vecteur texte et le vecteur image du même article.
**Sortie :** un vecteur de **6 068 nombres** (4 532 + 1 536).

L'opération semble triviale — on met les deux bouts à la suite — mais elle demande une précaution. Les deux blocs n'ont ni la même échelle ni la même densité : le bloc texte est creux avec des valeurs comprises entre 0 et 1, le bloc image est dense avec des valeurs qui peuvent être beaucoup plus grandes. Concaténés tels quels, le bloc image écraserait le bloc texte, non parce qu'il est plus informatif mais parce qu'il est plus bruyant.

Chaque bloc est donc **normalisé** — ramené à une longueur de 1 — avant d'être collé à l'autre. Cette normalisation se fait ligne par ligne, article par article, **sans aucun paramètre appris**. C'est ce qui garantit qu'elle ne peut pas transporter d'information du jeu de test vers le jeu d'entraînement. Un test automatisé vérifie cette propriété : normaliser un sous-ensemble donne exactement le même résultat que normaliser tout le jeu puis extraire le sous-ensemble.

## 7.3 Les têtes de classification

Deux têtes, choisies pour leurs propriétés opposées.

**XGBoost** construit une forêt d'arbres de décision successifs, chacun corrigeant les erreurs du précédent. Il est robuste sans réglage, il tolère les variables inutiles, et il indique quelles dimensions ont pesé. Sa faiblesse ici est structurelle : il se débrouille moins bien en très grande dimension creuse, où chaque arbre ne peut regarder qu'une poignée de termes à la fois.

**Le perceptron multicouche (MLP)** est un petit réseau de neurones — deux couches cachées de 128 puis 64 unités. Il combine toutes les dimensions à la fois, ce qui lui convient bien mieux ici. En contrepartie, c'est une boîte noire : on ne peut plus lire pourquoi il a décidé.

Il produit en sortie des nombres qui somment à 1 et qu'on lira comme une confiance. Une précision s'impose, parce que c'est un raccourci fréquent : **rien ne garantit que ces nombres soient calibrés**. Une sortie de 0,80 ne signifie pas « 80 % de chances d'avoir raison ». Ce qu'on leur demande est plus modeste — qu'ils **ordonnent** correctement les cas sûrs et les cas douteux. C'est suffisant pour fixer un seuil d'abstention, et c'est vérifié empiriquement en §8.5 plutôt que supposé.

**En entrée :** un vecteur, quelle qu'en soit l'origine.
**En sortie :** sept probabilités. La plus haute donne la catégorie ; **sa valeur donne la confiance**. Ce second nombre est aussi important que le premier — c'est lui qui rendra l'abstention possible.

## 7.4 Le protocole de comparaison

Six approches sont comparées. Pour que la comparaison veuille dire quelque chose, tout ce qui n'est pas l'objet de la comparaison est tenu constant : même découpe des données, mêmes 158 articles de test, même graine aléatoire, même tête quand c'est la représentation qu'on compare.

Une seule chose a demandé une correction en cours de route, et elle mérite d'être signalée. La première version du code mesurait le temps d'inférence sur des vecteurs **déjà calculés** — elle chronométrait la décision, pas la représentation. Les transformeurs affichaient alors 0,00 ms par article, ce qui est absurde : leur coût est presque entièrement dans l'encodage. Le chronomètre a été déplacé pour englober toute la chaîne, du texte brut à la prédiction. Sans cette correction, les modèles lourds étaient flattés précisément sur l'axe qui allait décider de la recommandation.

---

# 8. La comparaison

## 8.1 Quelle métrique, et pourquoi

Avant de mesurer, il faut décider **quoi** mesurer, et le décider avant de voir les résultats.

L'**exactitude** — la proportion de bonnes réponses — est la métrique intuitive, mais elle a un défaut : elle se laisse dominer par les grosses catégories. Ici, les classes étant équilibrées, elle reste honnête, et on la rapporte.

La **F1 macro** est la métrique retenue. Elle calcule, pour chaque catégorie, un équilibre entre deux erreurs de nature différente — ranger un article ailleurs qu'à sa place, et faire entrer dans une catégorie des articles qui n'y sont pas — puis fait la moyenne des sept, **sans pondérer par l'effectif**. Chaque catégorie compte autant. C'est ce qu'on veut : une petite catégorie mal servie ne doit pas disparaître dans la moyenne.

La **F1 de la catégorie la plus faible** est rapportée en plus, et c'est celle que je regarde en premier. Une moyenne masque toujours qui décroche.

À côté des métriques de qualité, trois métriques de coût : le temps d'entraînement, le **temps d'inférence de bout en bout**, et l'empreinte mémoire du modèle déployé.

**Le seuil d'acceptation a été fixé avant toute mesure : F1 macro ≥ 0,90.** Ce n'est pas un détail de méthode. Un seuil décidé après coup se cale toujours juste en dessous du résultat obtenu, et ne prouve alors plus rien.

## 8.2 Les résultats

| Modèle | F1 macro | Exactitude | F1 catégorie la plus faible | Entraînement | Inférence | Empreinte |
|---|---|---|---|---|---|---|
| **Fusion texte + image** | **0,974** | 0,975 | **0,955** | 1,8 s | 35,8 ms | 365 Mo |
| TF-IDF + MLP | 0,943 | 0,943 | 0,870 | 0,5 s | **0,06 ms** | **14 Mo** |
| DINOv2 figé — image seule | 0,937 | 0,937 | 0,864 | 0,2 s | 35,7 ms | 346 Mo |
| BERT figé (2018) | 0,924 | 0,924 | 0,870 | 17,3 s | 23,4 ms | 438 Mo |
| TF-IDF + XGBoost | 0,923 | 0,924 | 0,837 | 4,6 s | 0,08 ms | 2,6 Mo |
| ModernBERT figé (2024) | 0,885 | 0,886 | 0,800 | 31,6 s | 44,6 ms | 596 Mo |

Cinq modèles sur six franchissent le seuil de 0,90.

**C'est le résultat structurant de l'étude, et il est contre-intuitif :** puisque presque tout le monde réussit, la performance ne départage plus rien. Le critère de décision bascule sur le coût — et là, les écarts ne sont pas de quelques pour cent mais de plusieurs ordres de grandeur. Entre le modèle le plus léger et le plus lourd, il y a un facteur 230 en mémoire et un facteur 700 en temps de réponse.

## 8.3 Trois lectures du tableau

### La fusion gagne, et elle gagne là où ça compte

+3 points de F1 macro sur le meilleur modèle texte. Mais le chiffre à regarder est l'autre : **+8,5 points sur la catégorie la plus faible**, qui passe de 0,870 à 0,955.

C'est exactement ce qu'on espérait de l'image, et le mécanisme est intelligible. Les deux catégories qui décrochent chez tous les modèles texte sont *Home Decor & Festive Needs* et *Baby Care*. Un vase décoratif et un saladier se décrivent avec un vocabulaire voisin — matériau, dimensions, couleur — et le texte les confond. **Mais ils ne se ressemblent pas en photographie.** L'image apporte précisément le signal qui manquait, et pas ailleurs.

| Catégorie | Texte seul | Fusion | Gain |
|---|---|---|---|
| Home Decor & Festive Needs | 0,870 | 0,955 | **+8,5 pts** |
| Baby Care | 0,905 | 0,957 | +5,2 |
| Kitchen & Dining | 0,933 | 0,955 | +2,2 |
| Home Furnishing | 0,958 | 0,979 | +2,1 |
| Computers | 0,979 | 1,000 | +2,1 |
| Beauty and Personal Care | 0,977 | 0,977 | 0 |
| Watches | 0,978 | 1,000 | +2,2 |

L'image ne fait pas monter tout le monde uniformément : elle rattrape ceux qui étaient en retard. C'est le profil de gain qu'on veut.

La fusion coûte 600 fois le temps d'inférence du modèle texte. C'est énorme en relatif, et négligeable en absolu : 35,8 ms restent très en dessous du budget de 200 ms qu'impose l'expérience de publication. **La contrainte qui aurait pu écarter la fusion ne mord pas.** C'est pour cette raison, et pas seulement pour ses 3 points, qu'elle est retenue.

### Le plus récent et le plus lourd est le seul à échouer

ModernBERT (2024) est plus gros que BERT (2018), plus récent, meilleur sur à peu près tous les tests de référence publiés. Ici, il fait **4 points de moins**, coûte 36 % de mémoire en plus, et est le seul modèle sous le seuil.

Il n'y a pas de mystère, et l'explication est la limite déjà énoncée : ces modèles sont utilisés figés. Un modèle de langage masqué n'a jamais été entraîné pour que la moyenne de ses jetons soit une bonne représentation de phrase. Rien ne garantit qu'un meilleur modèle produise une meilleure moyenne. **Un classement établi sur des modèles réglés finement ne se transpose pas à un usage figé.**

### Le verdict dépend de la modalité — et c'est un résultat, pas un accident

Côté texte, la représentation lexicale (0,943) bat les deux encodeurs pré-entraînés (0,924 et 0,885), sur tous les axes à la fois : mieux, plus vite, plus léger.

Côté image, l'encodeur pré-entraîné figé atteint 0,937 **à lui seul**, mieux que tous les modèles de texte pré-entraînés.

La différence n'est pas la modalité, c'est l'objectif d'entraînement. DINOv2 est entraîné en auto-supervision précisément pour produire de bonnes représentations figées, et il tient sa promesse. Un modèle de langage masqué dont on moyenne les jetons n'a pas été entraîné pour ça, et il déçoit.

**« Encodeur pré-entraîné figé » n'est donc pas une catégorie homogène.** La bonne question à poser devant un modèle n'est pas « est-il récent ? » mais « pour quel usage a-t-il été entraîné ? ».

Sur le texte, la conclusion défendable reste étroite et il faut la garder étroite : à protocole constant, sur cette tâche et à ce volume, une représentation lexicale l'emporte. La tâche est lexicalement séparable — le vocabulaire d'une fiche produit est extrêmement discriminant — ce qui est précisément le terrain de jeu du TF-IDF. Sur des textes courts, ambigus ou multilingues, le classement s'inverserait probablement.

## 8.4 Un plafond, pas une performance

L'exactitude de la fusion s'établit à 0,9747, soit **quatre erreurs sur 158 articles**.

Ce chiffre mérite une lecture prudente plutôt qu'une célébration. On a établi en §4.4 que la vérité de référence est déclarative, donc bruitée. Quand le modèle contredit l'étiquette, on ne sait pas qui a tort. Il est donc raisonnable de penser qu'**une partie des quatre « erreurs » restantes n'en sont pas** — et symétriquement, qu'on ne pourra pas monter beaucoup plus haut, quel que soit le modèle, parce que la limite n'est plus dans le modèle mais dans l'étiquetage.

C'est ce qui motive la première recommandation de la §13 : à ce niveau, améliorer le modèle ne sert plus à rien. C'est l'étiquetage qu'il faut améliorer.

## 8.5 Savoir se taire : le seuil de confiance

Voici la partie de l'étude qui transforme un classifieur en système utilisable.

Le modèle ne renvoie pas seulement une catégorie, il renvoie sept probabilités. La plus haute mesure sa confiance. On peut donc décider : **en dessous d'un certain niveau, on n'affiche rien et l'article part en révision humaine.**

On perd en couverture ce qu'on gagne en fiabilité. La question est de savoir où placer le curseur, et ça se mesure :

| Seuil | Fusion — couverture | Fusion — erreurs | Texte seul — couverture | Texte seul — erreurs |
|---|---|---|---|---|
| 0,50 | 92,4 % | 1,4 % | 91,1 % | 2,8 % |
| **0,60** | **82,9 %** | **0 sur 131** | 84,2 % | 2,3 % |
| 0,70 | 71,5 % | 0 sur 113 | 76,6 % | 0,8 % |
| 0,80 | 62,0 % | 0 sur 98 | 67,7 % | 0 sur 107 |

**Le seuil recommandé est 0,60 sur la fusion.** À ce niveau, le modèle range 83 % du catalogue sans une seule erreur observée, et transmet les 17 % restants à un humain.

Le tableau montre aussi ce que l'image achète vraiment. Le modèle texte doit monter jusqu'à 0,80 pour atteindre zéro erreur, et il ne couvre alors plus que 68 %. **L'image achète 15 points de couverture à qualité égale** — c'est-à-dire 15 % du catalogue en moins à faire relire par un humain. C'est un argument bien plus concret que trois points de F1.

Une formulation s'impose ici, et c'est une question d'honnêteté statistique : on écrit « aucune erreur sur les 131 articles concernés », jamais « 100 % de précision ». Sur 131 observations, l'absence d'erreur constatée est compatible avec un vrai taux d'erreur de l'ordre de 2 %. Le chiffre rond mentirait.

---

# 9. L'optimisation

Jusqu'ici, tous les modèles tournaient avec des réglages choisis a priori. Il reste à savoir si ces réglages étaient les bons — et, plus utile encore, **de combien le résultat en dépend**.

## 9.1 Où l'on optimise, et surtout où l'on n'optimise pas

C'est ici que servent les 157 articles de validation, restés fermés depuis la §5.6.

Chaque combinaison de réglages est **entraînée sur les 735 articles d'entraînement, et jugée sur les 157 de validation**. Le jeu de test reste fermé pendant toute cette phase. Il ne sera rouvert qu'à la toute fin, une fois, avec la seule configuration retenue.

Cette discipline n'est pas cosmétique. Si l'on choisissait les réglages en regardant le score de test, ce score cesserait de mesurer la performance du modèle pour mesurer la qualité de notre sélection — un chiffre optimiste, non reproductible, et invérifiable de l'extérieur. C'est l'erreur méthodologique la plus commune, et la plus difficile à détecter dans le travail d'autrui.

## 9.2 Les hyperparamètres explorés

Trois familles, choisies parce que ce sont les trois endroits où une décision arbitraire avait été prise.

**La représentation — de quoi est fait le vocabulaire.**

| Réglage | Valeurs testées | Ce qu'il contrôle |
|---|---|---|
| `max_features` | 5 000 · 10 000 · tout | Combien de termes le vocabulaire retient. Le plafond initial coupait-il de l'information utile ? |
| `ngram_range` | (1,1) · (1,2) | Les bigrammes apportent-ils vraiment quelque chose, ou seulement du volume ? |
| `min_df` | 1 · 2 | À partir de combien d'occurrences un terme entre au vocabulaire. À 1, les termes uniques entrent — risque de surapprentissage. |

**La tête de classification — la forme du réseau et sa discipline.**

| Réglage | Valeurs testées | Ce qu'il contrôle |
|---|---|---|
| `hidden_layer_sizes` | (128,64) · (256,) · (512,256) | La capacité du réseau. Plus grand n'est pas mieux sur 735 exemples. |
| `alpha` | 10⁻⁴ · 10⁻³ · 10⁻² | La régularisation L2 — la pénalité imposée aux poids trop grands. Sur peu de données et beaucoup de dimensions, c'est le levier le plus important. |

**Le poids du bloc image — l'hyperparamètre caché de la fusion.**

C'est celui que je trouve le plus intéressant, parce qu'il n'était pas visible. En normalisant les deux blocs à la même longueur (§7.2), on leur a donné implicitement **le même poids**. Rien ne justifiait ce choix : c'était une conséquence de l'implémentation, pas une décision. On teste donc explicitement des poids de 0,25 à 4 appliqués au bloc image, pour savoir si le texte devrait dominer, ou l'inverse.

Un réglage qui n'apparaît nulle part dans le code est un réglage qu'on n'a pas choisi. Le rendre explicite est la moitié du travail d'optimisation.

## 9.3 Le classement dit une chose, le test en dit une autre

108 combinaisons ont été entraînées et jugées sur les 157 articles de validation. Voici le résultat, et il n'est pas celui qu'on espérait :

| | F1 sur la validation | F1 sur le test |
|---|---|---|
| Configuration initiale, choisie a priori | 0,918 | **0,943** |
| « Meilleure » configuration selon la validation | **0,943** | 0,917 |

**Le classement s'est exactement inversé**, et l'écart est le même des deux côtés : 2,5 points. La configuration que l'exploration avait désignée comme la meilleure fait moins bien, sur le test, que celle qu'on avait choisie sans rien explorer.

La même chose se produit sur la fusion. Le poids d'image gagnant sur la validation est 0,25 ; il donne 0,969 sur le test, contre **0,974** pour le poids de 1,0 qui était le réglage implicite de départ. *(La comparaison est même sévère envers la version optimisée, puisqu'elle hérite aussi de la représentation textuelle sélectionnée juste au-dessus : une mauvaise sélection se propage d'un étage à l'autre.)*

L'explication tient en une division. **Le jeu de validation contient 157 articles : un seul article vaut 0,64 point de F1.** Un écart de 2,5 points, c'est quatre articles. On a sélectionné une configuration sur quatre articles de différence — c'est-à-dire sur rien.

Ce n'est pas un accident d'exécution, c'est la propriété d'un jeu de sélection trop petit : plus on essaie de configurations, plus on a de chances d'en trouver une qui tombe juste par hasard. Explorer davantage aggrave le problème au lieu de le résoudre.

## 9.4 Ce qui compte vraiment : lire les moyennes, pas le classement

La bonne lecture d'une exploration n'est pas sa première ligne. C'est la performance moyenne de chaque valeur, **tous les autres réglages confondus** — ce qui moyenne précisément le bruit qui vient de fausser le classement.

**Côté texte :**

| Hyperparamètre | Valeur | F1 moyenne | Écart sur l'axe |
|---|---|---|---|
| `ngram_range` | **(1,2) — avec bigrammes** | **0,930** | **1,9 pt** |
| | (1,1) — unigrammes seuls | 0,911 | |
| `min_df` | **1** | 0,925 | 1,0 pt |
| | 2 | 0,915 | |
| `hidden_layer_sizes` | (128, 64) | 0,922 | 0,5 pt |
| | (512, 256) | 0,921 | |
| | (256,) | 0,917 | |
| `max_features` | 5 000 | 0,921 | 0,2 pt |
| | illimité | 0,920 | |
| | 10 000 | 0,919 | |
| `alpha` | 10⁻⁴ | 0,9201 | **0,03 pt** |
| | 10⁻³ | 0,9201 | |
| | 10⁻² | 0,9198 | |

Trois enseignements, et ils sont nets.

**Les bigrammes sont le seul réglage qui compte.** Ils apportent près de 2 points en moyenne, et la pire configuration de toute l'exploration — 0,866, sept points sous les autres — est celle qui combine unigrammes seuls et seuil d'entrée à 2, ce qui réduit le vocabulaire à 1 830 termes. Cela confirme l'intuition de la §5.4 : sur des fiches produit, l'information est dans les paires de mots (*baby care*, *hair dryer*, *bed linen*) autant que dans les mots.

**Le plafond de vocabulaire ne servait à rien.** 5 000 termes, 10 000 termes ou aucune limite (19 319 termes) donnent le même résultat à 0,2 point près. Le corpus ne contient tout simplement pas assez de termes discriminants pour que le plafond morde. Un réglage qu'on croyait important ne l'est pas — c'est une information utile, parce qu'elle dit où ne pas passer de temps.

**La régularisation ne fait rien du tout** — 0,03 point sur deux ordres de grandeur de variation. Ce n'est pas que le surapprentissage n'existe pas ici : c'est qu'un autre mécanisme s'en occupe déjà. Le réseau utilise l'**arrêt anticipé**, qui interrompt l'apprentissage dès que la performance interne cesse de progresser. Il intervient bien avant qu'`alpha` ait la moindre influence. On avait deux régularisations, une seule agit — et c'est celle qu'on n'avait pas réglée.

**Côté fusion**, le résultat est le plus intéressant du lot :

| Poids du bloc image | F1 moyenne | Dispersion |
|---|---|---|
| 0,25 | 0,940 | 3,7 pt |
| 0,50 | 0,942 | 1,3 pt |
| **1,00 — le réglage implicite** | **0,947** | **0,6 pt** |
| 2,00 | 0,936 | 1,3 pt |
| 4,00 | 0,918 | 2,0 pt |

Le poids de 1,0 est ici à la fois le meilleur en moyenne et le plus stable — sa dispersion est six fois plus faible que celle du poids qui avait gagné le classement. Le classement désignait 0,25, la moyenne désigne 1,0.

Une conclusion tient sans réserve : **laisser l'image dominer coûte cher.** À poids 4, on perd 2,9 points. Le texte reste le signal principal et l'image un complément, ce qui rejoint la §8.3 où l'image répare des catégories précises plutôt qu'elle n'améliore tout.

Le départage entre 0,25, 0,5 et 1,0 est en revanche à prendre avec prudence, et la §9.5 montre pourquoi : la validation croisée place le maximum ailleurs. Il fallait tester pour le savoir — un réglage qu'on n'a pas choisi n'est pas un réglage qu'on a validé — mais tester n'a pas suffi à trancher.

## 9.5 La correction : sélectionner par validation croisée

Le diagnostic de la §9.3 appelle une correction de méthode, pas un abandon.

Plutôt que de juger chaque configuration sur un unique paquet de 157 articles, on la juge **cinq fois, sur cinq découpes différentes** des 892 articles d'entraînement et de validation réunis. Chaque article sert une fois à évaluer et quatre fois à entraîner. On obtient ainsi une moyenne — plus stable — *et* un écart-type, qui dit enfin si un écart entre deux configurations veut dire quelque chose.

Une précaution s'impose dans cette découpe : **le vectoriseur est réajusté à l'intérieur de chaque bloc**, sur sa seule partie d'entraînement. Le construire une fois sur l'ensemble ferait entrer dans le vocabulaire des termes issus des articles qu'on s'apprête à évaluer — une fuite discrète, invisible dans les résultats, et qui gonflerait tous les scores de la même façon.

Le jeu de test, lui, reste fermé jusqu'au bout.

### Ce que la validation croisée montre d'abord : presque rien ne départage

72 configurations distinctes ont été évaluées cinq fois chacune. Deux nombres résument le résultat, et c'est leur rapport qui compte :

| | |
|---|---|
| Meilleure configuration | **0,945 ± 0,014** |
| Étendue de tout le classement | 3,5 points |

L'écart-type d'une configuration entre ses cinq blocs est de 1,4 point. L'étendue de tout le classement, du premier au dernier, est de 3,5 points — soit deux écarts-types et demi. Conséquence directe : **35 des 72 configurations sont à moins d'un écart-type du sommet, et 69 à moins de deux.**

La quasi-totalité de la grille est indiscernable. Ce n'est pas un échec de l'exploration, c'est sa réponse : à ce volume de données, ces réglages ne font pas de différence mesurable.

C'est précisément l'information que la validation simple était incapable de fournir. Elle produisait un classement d'apparence nette, sans aucun moyen de savoir que ses écarts n'existaient pas.

### Et la configuration qu'elle retient est celle du départ

| | Représentation | Tête |
|---|---|---|
| Configuration initiale, choisie a priori | 5 000 termes, bigrammes, seuil 2 | (128, 64), α = 10⁻⁴ |
| Retenue par validation croisée | 5 000 termes, bigrammes, seuil 2 | (128, 64), α = 10⁻⁴ |

Elles sont identiques. Le meilleur résultat de l'exploration reproduit les réglages qui avaient été posés au jugé.

Sur le test, cette configuration donne 0,937, contre 0,943 mesuré au §8.2. Les hyperparamètres étant les mêmes, l'écart ne vient pas d'eux : il vient de ce que le modèle final est ici réentraîné sur les 892 articles ayant servi à la sélection plutôt que sur les 735 de l'entraînement seul, ce qui reconstruit aussi le vocabulaire. **Six dixièmes de point sur 158 articles, c'est un article.** Il n'y a rien à en conclure, et c'est exactement ce que la §9.3 aurait dû nous empêcher de faire dire aux chiffres.

### Ce qui ressort quand même : la représentation, et elle seule

Une précaution de lecture s'impose ici, et elle a une conséquence pratique. Les moyennes marginales ne sont valables que sur un plan équilibré. Après déduplication de la grille, elles ne le sont plus pour la représentation : « vocabulaire illimité » ne survit que là où il change quelque chose — c'est-à-dire uniquement sur les configurations à bigrammes — si bien que sa moyenne est contaminée par l'effet des bigrammes. On compare donc les huit représentations directement, chacune portant le même nombre de configurations de tête.

| Représentation | Vocabulaire | F1 moyenne |
|---|---|---|
| **Bigrammes**, seuil d'entrée 1 | 22 626 | **0,937** |
| Bigrammes, seuil 1 | 10 000 | 0,935 |
| Bigrammes, seuil 1 | 5 000 | 0,933 |
| Bigrammes, seuil 2 | 5 000 – 5 412 | 0,929 |
| Unigrammes, seuil 1 | 5 000 – 5 275 | 0,925 |
| Unigrammes, seuil 2 | 2 129 | 0,920 |

Le classement est parfaitement ordonné, et il tient en une phrase : **les bigrammes d'abord, puis autant de vocabulaire que possible.** Les cinq premières lignes sont toutes des bigrammes, les trois dernières des unigrammes, sans un seul croisement. L'écart total est de 1,8 point — moins spectaculaire que les 7,7 points de la validation simple, parce que la moyenne sur cinq blocs a effacé le bruit qui les gonflait.

Du côté de la tête, où le plan reste équilibré, les moyennes marginales confirment ce qu'on avait vu : **0,42 point d'écart** entre les trois formes de réseau testées, **0,19 point** entre les trois valeurs de régularisation. Sur des écarts-types de 1,4 point, ces différences n'existent pas.

### Le poids de l'image : deux méthodes, deux réponses, et ce qu'il faut en retenir

Le même traitement appliqué à la fusion donne un résultat plus embarrassant, et il vaut d'être rapporté tel quel.

| Poids du bloc image | F1 moyenne (validation croisée) |
|---|---|
| 0,50 | 0,936 |
| 0,25 | 0,932 |
| 1,00 | 0,928 |
| 2,00 | 0,919 |
| 4,00 | 0,916 |

La validation simple plaçait 1,0 en tête ; la validation croisée place 0,5. Les deux méthodes se contredisent — sur 0,8 point d'écart, pour un écart-type de 1,6 point. **Elles ne se contredisent donc pas vraiment : elles disent toutes les deux que ces trois valeurs sont équivalentes.** Sur 45 configurations, 18 sont à moins d'un écart-type du sommet.

Ce sur quoi elles s'accordent, en revanche, est net et monotone : au-delà d'un poids de 1, la performance chute sans ambiguïté. Laisser l'image peser plus lourd que le texte dégrade le modèle. C'est la seule conclusion que ces mesures autorisent, et elle suffit.

Reste le test, qui apporte un argument que la F1 macro ne montrait pas. La configuration retenue par validation croisée — poids 0,5 — obtient **0,975 sur le test, exactement comme le poids de 1,0** retenu au §8.2. Mais sa catégorie la plus faible tombe à **0,913, contre 0,955**. À performance moyenne identique, elle abandonne quatre points sur le rayon le moins bien servi.

C'est précisément ce que la §8.1 annonçait en choisissant de suivre la classe la plus faible plutôt que la moyenne seule. Le poids de 1,0 est conservé — non parce qu'il gagne en moyenne, mais parce qu'il ne sacrifie personne.

## 9.6 Ce que l'exercice apprend

**L'optimisation n'a rien amélioré, et c'est un résultat à part entière.** La validation croisée a retenu, à l'identique, les réglages posés au jugé avant toute mesure. Un rapport qui n'aurait montré qu'un tableau final aurait pu laisser croire à un travail de réglage fructueux ; savoir que la grille explorée ne contenait rien de mieux vaut mieux qu'un gain inventé.

Ce n'est pas non plus du temps perdu. On sait maintenant **où il est inutile de chercher**, ce qui est précisément ce qu'on demande à une exploration lorsqu'elle ne trouve rien.

**On lit une exploration par ses moyennes, jamais par sa première ligne.** La première ligne d'un classement est la configuration qui a eu le plus de chance sur le jeu de sélection — c'est structurel, le maximum d'un échantillon bruité est biaisé vers le haut. Les deux passages le montrent : la validation simple désignait un gagnant qui perdait sur le test, et la validation croisée place 35 configurations sur 72 à moins d'un écart-type du sommet.

**La représentation décide, le classifieur suit.** L'écart imputable aux réglages de la tête est de 0,42 point ; celui imputable au choix de la représentation, de 1,8 point — quatre fois plus, et la validation simple donnait la même direction plus fort encore. Le piège est confortable : régler un réseau est ce qu'on sait faire, tandis que choisir une représentation demande de comprendre les données. Le temps était mieux investi du second côté, et le §8.3 le confirme à plus grande échelle — c'est là que se jouent les vrais écarts.

**Deux régularisations, une seule agit.** L'arrêt anticipé rend le paramètre `alpha` sans effet mesurable. Un réglage qu'on croit tenir peut être neutralisé par un autre qu'on n'a pas remarqué ; on ne s'en aperçoit qu'en le faisant varier.

**Un jeu de 157 articles ne permet pas de sélectionner.** C'est la limite à retenir pour la suite. Sur un volume plus important, la sélection redeviendra discriminante, et cette exploration vaudra la peine d'être reprise — cette fois avec un espoir raisonnable d'y trouver quelque chose.

---

# 10. La mise à disposition

Un modèle qui vit dans un carnet de notes n'a aucune valeur. Cette section décrit ce qui existe aujourd'hui, et la forme que prendrait une mise en production.

## 10.1 Ce qui existe : une démonstration interactive

Une application web permet de saisir une description, de choisir une photographie, et de voir les trois modèles répondre côte à côte — texte seul, image seule, fusion. Chaque prédiction s'affiche avec ses sept probabilités et son niveau de confiance, et la catégorie réelle est signalée sur l'axe.

Son objet n'est pas de servir des requêtes, c'est de **rendre le comportement du modèle observable** : voir, sur un article précis, que le texte hésite entre deux rayons et que l'image tranche, vaut mieux que n'importe quel tableau de métriques pour convaincre une équipe métier.

Ce n'est pas la forme d'une mise en production : une application de ce type recharge son état à chaque interaction et ne tient pas la charge. C'est un outil de conviction, pas un service.

## 10.2 L'architecture proposée pour une mise en production

Un service HTTP, exposé en **FastAPI**, avec le modèle chargé une fois au démarrage et gardé en mémoire.

```
                            ┌──────────────────────────────┐
   Formulaire vendeur       │        Service FastAPI       │
   (description + photo) ──►│                              │
                            │  1. vectorisation TF-IDF     │
                            │  2. encodage DINOv2          │
                            │  3. fusion + normalisation   │
                            │  4. MLP → 7 probabilités     │
                            │  5. arbitrage sur le seuil   │
                            └──────────────┬───────────────┘
                                           │
                        confiance ≥ 0,60   │   confiance < 0,60
                    ┌──────────────────────┴──────────────────┐
                    ▼                                         ▼
          catégorie pré-remplie,                    file de révision
          modifiable par le vendeur                     humaine
```

**Un point d'entrée unitaire**, appelé pendant la saisie du vendeur :

```http
POST /categoriser
{ "description": "Cotton Printed King sized Double Bedsheet ...",
  "image_url":   "https://.../a1b2c3.jpg" }
```

```json
{ "categorie":   "Home Furnishing",
  "confiance":   0.87,
  "decision":    "suggerer",
  "alternatives": [ {"categorie": "Home Decor & Festive Needs", "p": 0.07},
                    {"categorie": "Kitchen & Dining",           "p": 0.03} ],
  "modalites":   ["texte", "image"],
  "modele":      "fusion-mlp@1.1.1+sha256:4f9c2e…",
  "duree_ms":    38 }
```

Quatre champs de cette réponse ne servent pas à la prédiction et sont pourtant les plus importants. `decision` dit explicitement au client s'il doit afficher ou router — la règle de seuil vit dans le service, pas dans chaque application appelante. `alternatives` permet de proposer un choix court plutôt qu'une arborescence entière quand la confiance est moyenne. `modalites` indique si la photographie a réellement pu être lue. `modele` identifie la version qui a répondu.

**Un point d'entrée par lot**, pour reprendre l'existant. Les millions d'articles déjà en ligne ne se traitent pas un par un : ils se traitent en tâche de fond, par paquets, la nuit, en écrivant les résultats dans une file de correction plutôt qu'en modifiant le catalogue directement.

**Trois exigences non négociables.**

*La version du modèle voyage avec chaque prédiction.* Une empreinte du fichier de modèle est renvoyée dans chaque réponse et écrite dans le journal. Sans ça, le jour où le comportement change, on ne peut ni dater ni expliquer.

*Le repli texte est prévu par construction.* Une photographie manquante, illisible ou trop lourde ne doit pas faire échouer la requête : le service bascule sur le modèle texte seul, le signale dans sa réponse, et applique le seuil plus élevé qui convient à ce modèle (0,80 plutôt que 0,60). Sur un catalogue réel, ce cas n'est pas marginal.

*Le taux d'abstention est surveillé comme un indicateur métier.* Il est stable tant que le catalogue ressemble à ce sur quoi le modèle a appris. S'il monte, quelque chose a changé — un nouveau type de vendeur, une catégorie émergente, un changement de format des photographies. **C'est le détecteur de dérive le moins cher qui existe**, et il ne demande aucune étiquette.

## 10.3 Le déploiement

Le service et son modèle sont empaquetés ensemble dans une image conteneur, versionnée avec le code. Deux répliques derrière un répartiteur suffisent au volume actuel. L'encodage des images est le seul poste qui justifierait un accélérateur graphique, et seulement pour les traitements par lot — en unitaire, le CPU tient largement le budget de 200 ms.

L'intégration continue du dépôt exécute déjà, à chaque modification : la vérification de style, les tests, et **un rejeu complet de la chaîne depuis un clone vierge** avec vérification que le seuil de 0,90 est toujours atteint. Un modèle qui ne passe pas cette barrière ne peut pas être publié.

---

# 11. Ce qu'on obtient au bout

Le livrable n'est pas un score, c'est un composant avec des conditions d'emploi. Voici sa carte d'identité.

| | |
|---|---|
| **Ce qu'il fait** | Attribue l'une des 7 catégories de premier niveau à un article |
| **Ce qu'il lit** | Sa description (texte libre anglais) et sa photographie |
| **Ce qu'il rend** | Une catégorie, une confiance, les alternatives, sa propre version |
| **Sa fiabilité** | F1 macro 0,974 sur 158 articles jamais vus ; 0,955 sur sa catégorie la plus faible |
| **Sa règle d'emploi** | Suggère au-dessus de 0,60 de confiance ; se tait en dessous |
| **Sa couverture** | 83 % des articles rangés seuls, 0 erreur observée sur ces 83 % |
| **Son coût** | 36 ms par article ; 365 Mo en mémoire, chargés une fois |
| **Son repli** | Modèle texte seul (0,943, 0,06 ms, 14 Mo) si la photographie manque |
| **Ce qu'il ne fait pas** | Descendre dans l'arborescence au-delà du premier niveau ; reconnaître une catégorie qu'il n'a jamais vue ; s'améliorer tout seul |

**Ce qui est démontré :** l'automatisation est possible, elle atteint un seuil fixé d'avance, et il existe un réglage sous lequel elle ne commet aucune erreur observée sur 83 % du volume.

**Ce qui ne l'est pas :** que ces chiffres tiennent sur le catalogue réel. Le jeu est équilibré, propre, et daté ; le catalogue ne l'est pas. Les recommandations de la §13 portent toutes sur cet écart.

---

# 12. Le coût à l'échelle

Les chiffres de la §8 sont mesurés sur 158 articles. La question qui décide vraiment est : que se passe-t-il à un million ?

## 12.1 Le coût de calcul

À 36 ms par article, un million d'articles représente environ **10 heures de calcul**, parallélisables sans difficulté. Sur du matériel loué, on parle de quelques dizaines d'euros pour un traitement complet du catalogue, et de moins que ça en régime nominal où seuls les nouveaux articles sont traités.

Le modèle texte seul, à 0,06 ms, traiterait le même million en **une minute**.

Ces coûts sont négligeables devant tout le reste. Ce n'est pas là qu'il faut regarder — et c'est en soi un résultat, parce que la crainte du coût d'inférence est souvent ce qui bloque ce genre de projet.

## 12.2 Le coût qui compte vraiment : la révision humaine

Voici le calcul qui porte la décision. Hypothèses posées explicitement, pour qu'on puisse les contester :

- catalogue d'un million d'articles à ranger ;
- 20 secondes pour qu'un opérateur catégorise un article à la main ;
- 15 € de l'heure chargés.

| Scénario | Articles traités par un humain | Heures | Coût |
|---|---|---|---|
| Tout à la main | 1 000 000 | 5 556 h | **≈ 83 000 €** |
| Modèle texte, seuil 0,80 | 323 000 (32,3 %) | 1 794 h | ≈ 27 000 € |
| **Fusion, seuil 0,60** | **171 000 (17,1 %)** | **950 h** | **≈ 14 000 €** |

L'automatisation économise environ **69 000 € par million d'articles**, et la fusion en apporte 13 000 de plus que le modèle texte — pour un surcoût de calcul de quelques dizaines d'euros. **Le rapport est de l'ordre de 1 à 300.** C'est ce rapport, et non les 3 points de F1, qui justifie de retenir le modèle le plus lourd.

Ces chiffres valent ce que valent leurs hypothèses. Ils sont donnés pour montrer l'ordre de grandeur et la structure du raisonnement, pas pour être inscrits dans un budget.

## 12.3 L'impact métier au-delà du coût direct

Trois effets ne se chiffrent pas en heures économisées et pèsent probablement plus lourd.

**Les articles retrouvent leur visibilité.** Un article bien rangé est un article trouvable. Le gain de chiffre d'affaires est difficile à estimer sans données de conversion, mais il porte sur des articles déjà en catalogue, sans coût d'acquisition supplémentaire.

**La friction de publication baisse.** Un vendeur qui valide une suggestion publie plus vite qu'un vendeur qui descend une arborescence. Sur un modèle de place de marché, le nombre d'articles publiés est une métrique de croissance directe.

**Le système s'améliore tout seul, gratuitement.** Chaque fois qu'un vendeur corrige une suggestion, il produit une donnée d'entraînement parfaitement étiquetée — étiquetée par quelqu'un qui, lui, sait vraiment ce qu'il vend. C'est le meilleur mécanisme de collecte imaginable, et il ne coûte rien. Encore faut-il l'avoir prévu dès le départ : c'est la raison pour laquelle le §3 insiste sur « suggérer, pas imposer ».

## 12.4 Les effets qu'on n'a pas chiffrés, et qu'il faut surveiller

Trois risques ne se voient dans aucune métrique de performance. Les ignorer serait le vrai défaut de ce rapport.

**Le biais s'auto-entretient.** Les catégories que le modèle sert le moins bien sont celles où il s'abstiendra le plus, donc celles qui partiront en révision humaine, donc celles dont on corrigera le moins d'exemples si personne ne s'en occupe. Une catégorie mal servie a tendance à le rester. La parade est de suivre **la F1 de la catégorie la plus faible**, jamais l'exactitude globale — un modèle peut gagner en moyenne tout en abandonnant un rayon entier.

**L'opérateur cesse de vérifier.** C'est le risque le plus banal et le plus documenté des systèmes d'assistance : au bout de quelques centaines de suggestions justes, le vendeur valide sans lire. Le système devient alors une attribution silencieuse, ce que le §3 avait explicitement exclu. Deux garde-fous : afficher la confiance plutôt que de la garder pour soi, et suivre le **taux de correction manuelle** — s'il tombe à zéro, ce n'est pas que le modèle est parfait, c'est que plus personne ne regarde.

**Aucune donnée personnelle n'entre, et cela doit le rester.** Le système ne lit que la description et la photographie d'un article commercial. C'est un choix, pas un hasard : il rend le traitement trivial du point de vue réglementaire. Toute évolution qui voudrait ajouter des signaux de profil vendeur ou de comportement d'achat sortirait de ce cadre et demanderait une analyse propre. Le contrôle à mettre en place est simple — une revue du contenu du corpus avant chaque nouvelle ingestion.

---

# 13. Ce qu'il faut faire ensuite

Cinq chantiers, dans l'ordre où je les mènerais.

## 13.1 Améliorer l'étiquetage avant d'améliorer le modèle

C'est la première priorité et elle est contre-intuitive.

À 0,974, le modèle est probablement au niveau du bruit de l'étiquetage (§8.4). Continuer à travailler le modèle dans ces conditions, c'est l'entraîner à reproduire des erreurs de vendeurs.

**Ce qu'il faut faire :** prendre les articles sur lesquels le modèle est confiant *et* en désaccord avec le vendeur, et les faire ré-étiqueter par quelqu'un du métier. Ce lot est petit — quelques dizaines d'articles — et c'est exactement là que se concentre l'information. On saura alors si les erreurs restantes sont celles du modèle ou celles du catalogue, et on ne peut rien décider d'autre avant de le savoir.

Un corollaire immédiat : sur les catégories les plus confondues, *Home Decor* et *Baby Care*, la frontière est peut-être ambiguë dans la nomenclature elle-même. Un bougeoir décoratif qui sert à table appartient-il à *Home Decor* ou à *Kitchen & Dining* ? Si deux humains ne sont pas d'accord, aucun modèle ne tranchera. **Cette question relève du produit, pas de la science des données**, et il faut la lui renvoyer.

## 13.2 Plus de données, et surtout des données récentes

735 articles d'entraînement, c'est peu. Deux directions distinctes :

**Le volume.** Le catalogue réel en contient des ordres de grandeur de plus. Les modèles pré-entraînés — qui ont déçu ici — deviennent nettement plus intéressants à mesure que le volume monte, parce que le réglage fin cesse d'être hors de portée.

**La fraîcheur.** L'échantillon est daté. Un vocabulaire produit vieillit vite : marques, technologies, tournures commerciales. Avant tout déploiement, il faut ré-entraîner sur un extrait récent et vérifier que les chiffres tiennent. C'est un préalable, pas une amélioration.

## 13.3 Régler finement un encodeur, pour trancher la question ouverte

La conclusion de la §8.3 est bornée par un choix : les encodeurs sont figés. Cette borne n'est pas satisfaisante, elle est seulement honnête.

Régler finement un modèle de la taille de BERT sur quelques dizaines de milliers d'articles est aujourd'hui à portée. C'est le seul moyen de savoir si la représentation lexicale gagne vraiment, ou si elle gagne seulement parce qu'on a désavantagé ses concurrents. **Tant que cette expérience n'est pas faite, le classement du §8.2 ne doit pas être cité hors de son contexte.**

## 13.4 Descendre dans l'arborescence

Sept catégories, c'est un premier niveau. La valeur métier réelle est plus bas : `Bed Linen` est plus utile que `Home Furnishing` pour un acheteur.

L'approche naturelle est **hiérarchique** — un modèle pour le premier niveau, puis un modèle par branche. Chaque sous-modèle n'a alors à distinguer que des articles déjà voisins, sur un problème plus petit. Cela demande beaucoup plus de données par feuille, ce qui renvoie au §13.2.

## 13.5 Exploiter mieux l'image

Deux pistes, par ordre de rapport valeur/effort.

**La segmentation du produit.** Les photographies de commerce sont sur fond blanc, mais pas toutes : certaines montrent l'article en situation, avec des objets parasites. Isoler le produit avant de l'encoder retirerait ce bruit. C'est un ajout modeste au coût d'inférence.

**Le réglage fin de l'encodeur d'images.** DINOv2 figé atteint déjà 0,937 seul. Il est raisonnable de penser qu'un réglage sur des photographies de commerce ferait mieux — mais cela demande du volume, et cela transforme un composant réutilisable en composant spécifique à maintenir. À n'envisager qu'après le §13.1, faute de quoi on optimiserait contre un étiquetage bruité.

---

# Annexes

## A. Rejouer l'étude

```bash
pip install -r requirements.txt
make all                     # benchmark et figures, une dizaine de secondes
make benchmark ENCODERS=1    # ajoute les encodeurs de texte pré-entraînés
make optimize                # exploration des hyperparamètres
make demo                    # application de démonstration
```

Versions épinglées, graine fixe, découpe définie dans un module unique. L'intégration continue rejoue la chaîne depuis un clone vierge à chaque modification et échoue si le seuil de 0,90 n'est plus atteint.

## B. Les carnets

| Carnet | Ce qu'il établit |
|---|---|
| `01_eda_etl` | Le gisement, ses défauts, le piège du champ `brand`, la découpe |
| `02_visualisation` | Ce que les données racontent avant tout modèle |
| `03_modele_texte` | TF-IDF, les deux têtes, les encodeurs figés |
| `04_modele_image` | DINOv2 figé, protocole d'évaluation linéaire |
| `05_modele_combine` | La fusion, sa normalisation, l'absence de fuite |
| `06_comparaison` | Le tableau final, le seuil de confiance, la recommandation |

## C. Figures et données produites

| Fichier | Contenu |
|---|---|
| `reports/fig1_cout.png` | Performance contre coût d'inférence, échelle logarithmique |
| `reports/fig2_f1_par_classe.png` | Performance par catégorie et par modèle |
| `reports/fig3_transformations.png` | La chaîne de transformation sur un article réel |
| `reports/fig4_donnees.png` | Équilibre des classes, distribution des longueurs |
| `reports/benchmark.csv` | Le tableau de résultats complet |
| `reports/f1_par_classe.json` | Performance détaillée par catégorie |
| `reports/optimisation_texte_cv.csv` | Les 72 combinaisons explorées, moyenne et écart-type sur cinq blocs |
| `reports/optimisation_fusion_cv.csv` | Idem pour le poids du bloc image |
| `reports/representations_cv.csv` | Les huit représentations comparées à plan équilibré |
| `reports/effets_marginaux_*_cv.csv` | Moyenne par valeur d'hyperparamètre — à ne lire que sur les axes de la tête |
| `reports/optimisation_cv.json` | Le score de test de la configuration retenue, ouvert une seule fois |

Le suffixe `_cv` désigne la sélection par validation croisée. Les mêmes fichiers en `_holdout` sont produits par `make optimize HOLDOUT=1` et documentent la méthode naïve critiquée en §9.3.

## D. Note sur les métriques

Aucune métrique n'est bonne dans l'absolu. Chacune répond à une question, et chacune ment sur autre chose.

| Métrique | Sa question | Quand elle ment |
|---|---|---|
| Exactitude | Quelle part de bonnes réponses ? | Dès que les classes sont déséquilibrées — ce n'est pas le cas ici |
| F1 macro | Toutes les catégories sont-elles traitées à égalité ? | C'est une moyenne : elle masque qui décroche |
| F1 par catégorie | Où se concentre l'erreur ? | Bruitée sur petits effectifs — 22 articles par catégorie au test |
| Temps d'inférence | Le passage à l'échelle tient-il ? | Ment si la vectorisation ou l'encodage sont hors du chronomètre |
| Empreinte | CPU ou accélérateur graphique ? | Ignore la mémoire vive réellement consommée à l'exécution |
| Couverture au seuil | Quel volume est automatisable ? | Sans son taux d'erreur associé, elle ne veut rien dire |

## E. Où trouver quoi

Le rapport suit le fil du raisonnement plutôt que le découpage d'une grille d'évaluation. Voici la correspondance, pour qui cherche un point précis.

| Ce qu'on cherche | Où c'est traité |
|---|---|
| Le besoin métier et son origine | §1 — le fonctionnement de la place de marché, ce qui casse, qui paie |
| Les acteurs et leurs attentes | §1.2 — vendeur, acheteur, plateforme, et le coût pour chacun |
| Le cadrage de la demande | §1.4 — les quatre questions posées, et celle qu'on oublie |
| Le positionnement de l'auteur | §2 — ce qui est à portée, ce qui ne l'est pas |
| Les cas d'usage retenus | §3 — suggérer à la publication, s'abstenir sous seuil |
| L'audit du gisement de données | §4 et §5 — volume, qualité, fuite identifiée, décisions d'ETL |
| L'exploration et la visualisation | §6 — quatre questions, quatre figures |
| Les approches comparées | §7 — entrées, sorties, dimensions, coût de chacune |
| Le protocole d'évaluation | §5.6 et §7.4 — découpe en trois, chronomètre de bout en bout |
| Les métriques et leur choix | §8.1 et annexe D — ce que chacune mesure, et ce sur quoi elle ment |
| Les résultats comparés | §8.2 et §8.3 |
| L'arbitrage confiance / couverture | §8.5 — le seuil, et ce qu'il achète |
| L'optimisation des hyperparamètres | §9 — ce qui compte, ce qui ne compte pas, et pourquoi la méthode naïve échoue |
| L'architecture de mise en production | §10.2 — service, repli, traçabilité, surveillance |
| Les indicateurs de pilotage | §11 — la carte d'identité du modèle et ses conditions d'emploi |
| Les coûts et l'impact métier | §12 — calcul, révision humaine, effets non chiffrés |
| Les risques et leur maîtrise | §12.4 et annexe G |
| Les essais abandonnés et leurs causes | annexe F |
| La suite du travail | §13 — cinq chantiers, dans l'ordre |

## F. Ce qui a été tenté puis abandonné

Un rapport qui ne montre que ce qui a marché ne montre pas le travail. Voici les six décisions qui ont été prises puis défaites, avec ce qui les a fait tomber.

**Le chronomètre était au mauvais endroit.** La première version mesurait le temps d'inférence sur des vecteurs déjà calculés. Les transformeurs affichaient 0,00 ms par article — un chiffre absurde qui n'a pas sauté aux yeux immédiatement, parce qu'il allait dans le sens de ce qu'on attendait d'eux. Le chronomètre englobe désormais toute la chaîne. Sans cette correction, les modèles lourds étaient flattés précisément sur l'axe qui portait la recommandation (§7.4).

**Le champ `brand` a d'abord été conservé.** Il paraissait informatif. C'est l'examen de la *répartition* de ses valeurs manquantes, et non de leur nombre, qui a montré qu'il fuitait. Écarté (§5.3).

**La démonstration empilait les barres.** Les probabilités des trois modèles étaient superposées, ce qui les additionnait visuellement : une catégorie où le texte donnait 0,17 et la fusion 0,75 s'affichait à 0,92 — un nombre qui n'existe pas. Remplacé par des barres groupées. C'est le rappel que le choix d'une représentation graphique est une question de correction, pas de goût.

**Le repère de vérité terrain a été essayé en couleur.** Un doré, d'abord — mais il voisinait l'orange d'une des trois séries et les deux convergeaient pour un œil déficient au rouge-vert. Une bande de fond a ensuite cassé le calcul de position des barres groupées. La solution retenue n'utilise pas la couleur du tout : un chevron, une graisse, un contraste. Trois canaux, aucun conflit possible avec les séries.

**La sélection d'hyperparamètres sur le jeu de validation a été abandonnée.** Elle désignait des configurations qui perdaient sur le test. Remplacée par une validation croisée à cinq blocs (§9.3 et §9.5). La méthode naïve est conservée dans le code, derrière une option, parce que sa comparaison avec la bonne méthode est le résultat.

**Le poids du bloc image a été remis en cause, et le réglage de départ a tenu.** L'exploration a confirmé que donner le même poids aux deux blocs était le bon choix — mais on ne le savait pas avant de l'avoir mesuré, et le laisser implicite aurait été le laisser non validé (§9.4).

## G. Limites de l'étude, rassemblées

Elles sont dispersées dans le texte ; les voici au même endroit.

1. **L'étiquetage est déclaratif** et donc bruité. Toute performance est mesurée contre une référence imparfaite (§4.4).
2. **Les encodeurs sont figés**, jamais réglés finement. On compare des représentations, pas des modèles (§7.2).
3. **Le jeu est parfaitement équilibré**, ce que le catalogue réel n'est pas (§4.3).
4. **Le corpus est daté.** Le vocabulaire produit vieillit (§13.2).
5. **Seul le premier niveau de l'arborescence** est traité (§4.3).
6. **Les effectifs de test sont petits** — 158 articles, 22 par catégorie. Les écarts de moins de 2 points ne sont pas significatifs.
7. **Les chiffres de coût de la §12** reposent sur des hypothèses posées par l'auteur, pas sur des données Flipkart.
