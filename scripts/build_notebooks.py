"""Construit les carnets qui suivent la mission, puis les exécute.

Les carnets 01 et 02 — exploration et visualisation — restent valables tels
quels. Les suivants sont reconstruits ici pour suivre l'ordre de la mission :
représenter, projeter et mesurer, classer, collecter.

    python scripts/build_notebooks.py           # construit et exécute
    python scripts/build_notebooks.py --sec     # construit sans exécuter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
CARNETS = ROOT / "notebooks"

EN_TETE = """import sys
sys.path.insert(0, "..")
import numpy as np
import pandas as pd
pd.set_option("display.width", 160)"""


def md(texte: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(texte.strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


# --------------------------------------------------------------------------- 03

REPRESENTATIONS = [
    md("""
# 03 · Représenter le texte et les images

**Ce que fait ce carnet.** Un algorithme ne compare pas des mots ni des pixels : il compare des
nombres. Ce carnet applique les sept représentations demandées par la mission — cinq pour le texte,
deux pour l'image — et montre ce que chacune produit sur un même article.

**Ce qu'il établit.** Chaque produit existe désormais sous sept formes numériques, de 256 à 5 000
dimensions. Le carnet suivant dira laquelle rapproche les produits d'une même catégorie.
"""),
    code(EN_TETE),
    md("""
## L'article que nous suivons

Une montre, dont la fiche tient en trente-quatre mots. Nous la garderons sous la main dans tous les
carnets, jusqu'à sa prédiction finale.
"""),
    code("""
from src.pipeline import LABEL_COL, TEXT_COL, load

df = load()
montre = df[df["product_name"].str.contains("V9 METAL STRAP", na=False)].iloc[0]

print(montre["product_name"])
print()
print(montre[TEXT_COL])
"""),
    md("""
## Du texte brut aux jetons

Minuscules, ponctuation retirée, mots-outils anglais écartés. Rien de plus : ni racinisation, ni
correction orthographique. Tronquer les mots détruirait des références de modèles, qui sont parfois
les termes les plus discriminants d'une fiche.
"""),
    code("""
from src.pretraitement import etapes_texte

etapes = etapes_texte(montre[TEXT_COL])
print(f"{len(etapes['mots'])} mots bruts → {len(etapes['jetons'])} jetons retenus")
print()
print(" · ".join(etapes["jetons"]))
"""),
    md("""
## Comptage simple, puis pondération

Le comptage est la référence la plus rudimentaire qu'on puisse construire : combien de fois chaque
mot apparaît. Le TF-IDF reprend ce comptage et le divise par la fréquence du terme dans tout le
corpus — un mot présent partout ne distingue rien.

L'effet se lit sur notre montre : `color`, en tête au comptage, disparaît des premiers rangs une
fois pondéré, parce qu'il apparaît dans presque toutes les fiches du catalogue.
"""),
    code("""
from sklearn.feature_extraction.text import CountVectorizer

from src.text import vectoriseur

textes = df[TEXT_COL].tolist()

cv = CountVectorizer(lowercase=True, stop_words="english", max_features=5000, min_df=2).fit(textes)
tf = vectoriseur().fit(textes)

def sommet(vecteur, noms, n=6):
    v = vecteur.tocoo()
    return [(noms[c], round(float(d), 3)) for c, d in sorted(zip(v.col, v.data), key=lambda t: -t[1])[:n]]

noms_cv = np.array(cv.get_feature_names_out())
noms_tf = np.array(tf.get_feature_names_out())

print(f"comptage : {len(noms_cv)} mots au vocabulaire")
print("  ", sommet(cv.transform([montre[TEXT_COL]]), noms_cv))
print(f"TF-IDF   : {len(noms_tf)} termes au vocabulaire")
print("  ", sommet(tf.transform([montre[TEXT_COL]]), noms_tf))
"""),
    md("""
## Les sept représentations

Word2Vec apprend un vecteur par mot sur notre seul corpus, BERT produit une représentation qui
dépend du contexte, USE représente la phrase entière. Côté image, SIFT décrit des points
remarquables regroupés en « mots visuels », et VGG16 — privé de sa couche de classification — sert
d'extracteur.

Tout est mis en cache : le premier calcul prend plusieurs minutes, les suivants sont immédiats.
"""),
    code("""
from src.representations import IMAGE, TEXTE, obtenir

entrees = {"texte": textes, "image": df["uniq_id"].tolist()}
formes = {}

for famille, registre in (("texte", TEXTE), ("image", IMAGE)):
    for nom in registre:
        X, secondes = obtenir(nom, entrees[famille])
        formes[nom] = X
        etat = f"{secondes:.0f} s" if secondes else "cache"
        print(f"  {nom:22s} {X.shape[0]} × {X.shape[1]:5d}  ({etat})")
"""),
    md("""
## Ce que la montre devient

La même fiche, traduite sept fois. Les représentations lexicales sont creuses — la quasi-totalité
des dimensions vaut zéro — là où les représentations denses n'ont aucune valeur nulle.
"""),
    code("""
i = int(np.where(df["uniq_id"].values == montre["uniq_id"])[0][0])

lignes = []
for nom, X in formes.items():
    v = X[i]
    lignes.append({
        "Représentation": nom,
        "Dimensions": X.shape[1],
        "Valeurs non nulles": int((v != 0).sum()),
        "Minimum": round(float(v.min()), 2),
        "Maximum": round(float(v.max()), 2),
    })

pd.DataFrame(lignes).sort_values("Dimensions", ascending=False)
"""),
    md("""
**Ce que ce carnet établit.** Sept représentations disponibles, calculées sur le même corpus et
mises en cache. Aucune étiquette n'est intervenue : elles ne serviront qu'à mesurer, dans le carnet
suivant.
"""),
]

# --------------------------------------------------------------------------- 04

FAISABILITE = [
    md("""
# 04 · Étude de faisabilité — projeter, regrouper, mesurer

**Ce que fait ce carnet.** Il répond à la question centrale de la mission : les produits d'une même
catégorie se rapprochent-ils spontanément, une fois traduits en nombres ? On projette en deux
dimensions pour regarder, puis on mesure l'accord entre des groupes formés sans étiquettes et les
vraies catégories.

**Ce qu'il établit.** L'information est bien présente. Les caractéristiques visuelles issues d'un
réseau pré-entraîné structurent le catalogue nettement mieux que le texte — et bien mieux que SIFT,
qui reste proche du hasard sur les mêmes photographies.
"""),
    code(EN_TETE),
    md("""
## Le protocole

L'étude est non supervisée : les catégories ne servent qu'à colorier les graphiques et à mesurer
l'accord final. Elles n'entrent jamais dans la construction des représentations, ce qui autorise à
travailler sur les 1 050 produits sans risque de fuite.

Une analyse en composantes principales ramène chaque représentation à 50 dimensions, puis t-SNE la
met en plan. Chaque produit est d'abord ramené à une longueur unitaire, faute de quoi une
description longue occuperait mécaniquement une position plus éloignée de l'origine.
"""),
    code("""
from src.faisabilite import etudier
from src.pipeline import LABEL_COL, TEXT_COL, load
from src.representations import IMAGE, TEXTE, obtenir

df = load()
categories = df[LABEL_COL].tolist()
entrees = {"texte": df[TEXT_COL].tolist(), "image": df["uniq_id"].tolist()}

etudes = {}
for famille, registre in (("texte", TEXTE), ("image", IMAGE)):
    for nom in registre:
        X, _ = obtenir(nom, entrees[famille])
        etudes[nom] = etudier(X, categories)
        print(f"  {nom:22s} ARI projection {etudes[nom]['ARI projection']:.3f}")
"""),
    md("""
## L'indice de Rand ajusté, et ce qu'il n'est pas

L'indice compare deux découpages d'un même ensemble. Il vaut 1 lorsqu'ils coïncident, 0 lorsque leur
accord n'excède pas le hasard. L'ajustement compte : avec sept groupes de tailles voisines, deux
découpages tirés au sort présentent déjà un accord apparent que l'indice brut compterait à tort.

**Un indice de 0,51 ne signifie pas que 51 % des produits sont bien classés.** Ce n'est pas une
proportion, et les groupes formés n'ont d'ailleurs pas de nom : rien ne dit lequel correspond aux
montres.

Nous rapportons la mesure deux fois — sur la projection, que nous avons regardée, et sur la
représentation complète, parce que t-SNE déforme et qu'il serait commode de ne retenir que le plus
flatteur des deux chiffres.
"""),
    code("""
tableau = pd.DataFrame([
    {
        "Représentation": nom,
        "Dimensions": e["dimensions"],
        "ARI (projection 2D)": e["ARI projection"],
        "ARI (espace complet)": e["ARI représentation complète"],
    }
    for nom, e in etudes.items()
]).sort_values("ARI (projection 2D)", ascending=False)

tableau
"""),
    md("""
## Ce que le tableau dit

**Sur les mêmes photographies**, VGG16 atteint 0,51 quand SIFT reste proche du hasard. SIFT décrit
des motifs locaux — un angle, une texture — utiles pour reconnaître qu'une même scène a été
photographiée deux fois, mais qui ne disent rien de ce qu'est l'objet.

**VGG16 est la seule représentation dont l'accord est meilleur avant réduction qu'après.** Sa
structure ne doit donc rien à t-SNE. Pour les représentations textuelles, l'écart va dans l'autre
sens : lue seule, la colonne de gauche leur accorderait une netteté que l'espace d'origine ne
confirme pas.

**Le comptage simple fait pratiquement jeu égal avec BERT.** Sur des fiches de spécifications, où le
vocabulaire est très discriminant et la syntaxe presque absente, comprendre le contexte n'apporte
presque rien de plus que compter les mots.
"""),
    code("""
meilleure = tableau.iloc[0]["Représentation"]
etude = etudes[meilleure]

croise = pd.crosstab(
    pd.Series(categories, name="catégorie réelle"),
    pd.Series(etude["groupes_projection"], name="groupe"),
)
print(f"{meilleure} — chaque catégorie a-t-elle son groupe ?")
croise
"""),
    md("""
## Là où ça résiste

*Home Furnishing* se scinde presque en deux : une moitié dans son groupe, l'autre dans celui de
*Baby Care*. Regardons ce que contient réellement cette seconde moitié.
"""),
    code("""
groupe_baby = int(croise.loc["Baby Care"].idxmax())
avec = df.assign(groupe=etude["groupes_projection"])

print("Home Furnishing tombés dans le groupe de Baby Care :")
for nom in avec[(avec[LABEL_COL] == "Home Furnishing") & (avec.groupe == groupe_baby)]["product_name"].head(6):
    print("   -", nom[:70])

print()
print("Baby Care de ce même groupe :")
for nom in avec[(avec[LABEL_COL] == "Baby Care") & (avec.groupe == groupe_baby)]["product_name"].head(6):
    print("   -", nom[:70])
"""),
    md("""
Des housses de coussin, des couettes et des tapis de bain d'un côté ; des serviettes en coton, des
pyjamas de bébé et des protège-matelas de l'autre. Ce groupe ne correspond à aucune des deux
catégories : il rassemble des **textiles imprimés photographiés à plat**.

L'algorithme a regroupé par matière et par mise en scène, ce qui est ce qu'on lui a demandé de
faire, alors que la nomenclature du site regroupe par usage commercial. Une couette et un pyjama de
bébé n'ont rien en commun pour un acheteur ; ils se ressemblent beaucoup pour un réseau de vision.

**Ce que ce carnet établit.** L'information nécessaire à la catégorisation est présente dans les
données, et suffisamment pour que des groupes cohérents émergent sans qu'aucune étiquette n'ait été
montrée. Les photographies traitées par un réseau pré-entraîné sont la source la plus prometteuse.
"""),
]

# --------------------------------------------------------------------------- 05

SUPERVISE = [
    md("""
# 05 · Classification supervisée des images

**Ce que fait ce carnet.** Puisque l'information est présente dans les photographies, que devient la
performance lorsqu'on entraîne réellement un modèle à prédire la catégorie ? Le carnet compare
plusieurs stratégies de data augmentation sur le jeu de validation, puis n'ouvre le jeu réservé
qu'une fois, avec la seule stratégie retenue.

**Ce qu'il établit.** 137 produits sur 158 correctement classés à partir des seules images. La data
augmentation n'apporte pas d'amélioration nette, mais elle déplace les erreurs d'une catégorie à
l'autre.
"""),
    code(EN_TETE),
    md("""
## Le protocole, et pourquoi il compte

Tant qu'on cherchait à savoir si des groupes existaient, tout le corpus pouvait servir. Il faut à
présent réserver des produits que le modèle ne verra pas pendant son apprentissage.

La découpe se fait en trois parts et non en deux. Comparer plusieurs stratégies directement sur le
jeu de test, puis retenir la meilleure, ferait de son score une mesure de la qualité de notre
sélection autant que de celle du modèle. Le jeu de validation existe pour absorber ces comparaisons.
"""),
    code("""
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.pipeline import LABEL_COL, load, split
from src.supervise_image import extraire, tete

df = load()
train, val, test = split(df)
enc = LabelEncoder().fit(df[LABEL_COL])
etiquettes = list(enc.classes_)
y_tr, y_va, y_te = (enc.transform(d[LABEL_COL]) for d in (train, val, test))
ids_tr, ids_va, ids_te = (d["uniq_id"].tolist() for d in (train, val, test))

print(f"entraînement {len(ids_tr)} · validation {len(ids_va)} · réservé {len(ids_te)}")
"""),
    md("""
## Le modèle

VGG16 conserve ses poids d'origine et sert d'extracteur ; seule une petite tête de classification
est apprise par-dessus. Réajuster les 138 millions de paramètres du réseau sur 735 images le
conduirait à apprendre ces images plutôt que la tâche.

Le fonctionnement tient en une phrase : la photographie entre dans le réseau figé, qui en produit
512 nombres ; la tête reçoit ces 512 nombres et rend 7 probabilités.
"""),
    code("""
X_tr, _ = extraire(ids_tr)
X_va, _ = extraire(ids_va)
X_te, _ = extraire(ids_te)
print(f"chaque image devient {X_tr.shape[1]} nombres")
"""),
    md("""
## La montre, une dernière fois

Elle appartient au jeu réservé : le modèle ne l'a jamais vue.
"""),
    code("""
montre = test[test["product_name"].str.contains("V9 METAL STRAP", na=False)]
X_montre, _ = extraire(montre["uniq_id"].tolist())

modele_simple = tete().fit(X_tr, y_tr)
probabilites = modele_simple.predict_proba(X_montre)[0]

for categorie, p in sorted(zip(etiquettes, probabilites), key=lambda t: -t[1]):
    print(f"  {categorie:28s} {p:.3f}")
"""),
    md("""
## La data augmentation

On fabrique de nouvelles images d'entraînement en transformant celles dont on dispose. Le choix des
transformations est contraint par la nature des photographies : des produits de catalogue, centrés
et cadrés de la même manière. Un retournement horizontal ou une légère rotation restent plausibles ;
un retournement vertical produirait des images qu'on ne rencontrera jamais.

Quatre stratégies sont comparées **sur la validation**, pas sur le jeu réservé.
"""),
    code("""
strategies = [
    ("Sans augmentation", "aucune", 0),
    ("Augmentation douce ×4", "douce", 4),
    ("Augmentation forte ×4", "forte", 4),
    ("Augmentation forte ×8", "forte", 8),
]

selection, par_classe, modeles = [], {}, {}
for nom, intensite, n in strategies:
    if n == 0:
        X, y = X_tr, y_tr
    else:
        X_aug, index = extraire(ids_tr, intensite=intensite, copies=n)
        X, y = np.vstack([X_tr, X_aug]), np.concatenate([y_tr, y_tr[index]])

    clf = tete().fit(X, y)
    pred = clf.predict(X_va)
    modeles[nom] = clf
    scores = f1_score(y_va, pred, average=None, labels=range(len(etiquettes)))
    par_classe[nom] = dict(zip(etiquettes, scores.round(3)))
    selection.append({
        "Stratégie": nom,
        "Images d'entraînement": int(X.shape[0]),
        "F1 macro (validation)": round(float(f1_score(y_va, pred, average="macro")), 4),
    })

pd.DataFrame(selection).sort_values("F1 macro (validation)", ascending=False)
"""),
    md("""
Sur la validation, l'augmentation douce obtient le meilleur score. Le gain — six millièmes de point
de F1 macro, soit **moins d'un produit sur 157** — est trop faible pour conclure à une amélioration
nette. Une augmentation plus forte et répétée dégrade en revanche nettement la performance.

Une explication possible tient à la nature très standardisée des photographies : multiplier les
transformations artificielles peut éloigner les images d'entraînement de la distribution réellement
observée. C'est une hypothèse plausible, que ces quatre essais ne démontrent pas.

Le résultat est plus intéressant catégorie par catégorie.
"""),
    code("""
pd.DataFrame(par_classe)
"""),
    md("""
*Baby Care*, la catégorie la plus fragile, gagne près de sept points à mesure que l'augmentation
s'intensifie. *Computers* et *Home Furnishing* suivent le chemin inverse. La moyenne ne bouge pas
parce que les gains et les pertes se compensent : **l'augmentation déplace les erreurs plutôt
qu'elle ne les supprime**.

Cela suggère qu'une augmentation générique, appliquée uniformément, n'est probablement pas la bonne
stratégie.

## Le jeu réservé, ouvert une fois
"""),
    code("""
retenue = max(selection, key=lambda r: r["F1 macro (validation)"])["Stratégie"]
pred_te = modeles[retenue].predict(X_te)

print(f"stratégie retenue : {retenue}")
print(f"F1 macro          : {f1_score(y_te, pred_te, average='macro'):.4f}")
print(f"exactitude        : {accuracy_score(y_te, pred_te):.4f}")
print(f"                    {int(accuracy_score(y_te, pred_te) * len(y_te))}/{len(y_te)} bien classés")
"""),
    code("""
confusion = pd.crosstab(
    pd.Series([etiquettes[i] for i in y_te], name="réelle"),
    pd.Series([etiquettes[i] for i in pred_te], name="prédite"),
)
confusion
"""),
    md("""
Deux *Baby Care* sont prédits *Home Furnishing*, et deux *Home Furnishing* sont prédits *Baby Care*.
C'est la confusion que le regroupement sans étiquettes avait déjà fait apparaître au carnet 04.

Sa réapparition dans deux approches très différentes suggère qu'elle ne dépend pas uniquement du
modèle employé, et qu'il existe une ambiguïté réelle entre certaines images de ces deux catégories.
La supervision la réduit sans la faire disparaître.

**Ce que ce carnet établit.** L'image seule porte une part importante de l'information : 137 produits
sur 158, avec un réseau dont aucun poids n'a été réentraîné et sans utiliser une ligne de
description.
"""),
]

# --------------------------------------------------------------------------- 06

COLLECTE = [
    md("""
# 06 · Collecte de nouveaux produits via une API

**Ce que fait ce carnet.** La marketplace envisage d'élargir sa gamme à l'épicerie fine. Avant tout
modèle, une question plus terre à terre : peut-on récupérer automatiquement ces produits, avec les
informations nécessaires ?

**Ce qu'il établit.** La collecte fonctionne, mais dix produits suffisent à révéler un enjeu de
qualité et d'homogénéité des métadonnées, à traiter en amont du modèle.
"""),
    code(EN_TETE),
    md("""
## La source et la correspondance des champs

Open Food Facts ne demande aucune inscription : le carnet reste exécutable par un tiers, sans clé à
transmettre. Cette base est alimentée de façon collaborative, ce qui aura son importance.

Les cinq champs attendus viennent du schéma d'Edamam, l'autre source proposée. Quatre
correspondances sont évidentes ; la cinquième demande un jugement — `foodContentsLabel` désigne la
composition d'un produit, dont `ingredients_text` est l'équivalent le plus proche.
"""),
    code("""
from collecte_api import CORRESPONDANCE, interroger, normaliser

pd.DataFrame(
    [{"Champ demandé": k, "Champ Open Food Facts": v} for k, v in CORRESPONDANCE.items()]
)
"""),
    md("""
## La collecte

Le filtre porte sur la catégorie et non sur le texte libre : une recherche plein texte remonterait
aussi tout ce qui mentionne le mot sans en être — vinaigres, sauces, arômes.
"""),
    code("""
produits = [normaliser(p) for p in interroger("champagne", 10)]
collecte = pd.DataFrame(produits)

print(f"{len(collecte)} produits collectés")
for champ in CORRESPONDANCE:
    print(f"  {champ:20s} renseigné pour {(collecte[champ] != '').sum()}/{len(collecte)}")
"""),
    code("""
collecte[["foodId", "label", "category"]]
"""),
    md("""
## Ce que la collecte apprend

Trois observations, sur dix produits seulement.

**Les catégories sont hétérogènes.** Certaines étiquettes n'ont pas été traduites et conservent un
préfixe de langue ; l'une d'elles ne dit à peu près rien du produit.
"""),
    code("""
from collections import Counter

etiquettes = Counter(c.strip() for ligne in collecte["category"] for c in ligne.split(","))
pd.Series(etiquettes).sort_values(ascending=False).to_frame("produits concernés")
"""),
    md("""
**Les libellés sont irréguliers**, trace visible de la saisie collaborative et de la reconnaissance
automatique d'étiquettes. **Et la catégorie source n'est pas toujours juste** — un cocktail à la
pêche figure parmi les champagnes.
"""),
    code("""
for ligne in collecte.itertuples():
    contenu = ligne.foodContentsLabel[:60] if ligne.foodContentsLabel else "— vide —"
    print(f"  {ligne.label[:52]:<52} | {contenu}")
"""),
    code("""
from pathlib import Path

sortie = Path("..") / "reports" / "produits_champagne.csv"
collecte.to_csv(sortie, index=False)
print(f"écrit dans {sortie}")
"""),
    md("""
**Ce que ce carnet établit.** La collecte automatique fonctionne et produit le fichier demandé. Elle
reste exploratoire — dix produits ne caractérisent pas une base entière — mais elle met déjà en
évidence un enjeu de qualité et d'homogénéité des métadonnées, qui devra être traité en amont du
modèle de classification. On retrouve, sous une autre forme, la difficulté du point de départ : ici
comme sur la marketplace, les catégories sont déclarées par des contributeurs qui ne suivent pas
tous la même règle.
"""),
]

CARNETS_A_CONSTRUIRE = {
    "03_representations.ipynb": REPRESENTATIONS,
    "04_faisabilite.ipynb": FAISABILITE,
    "05_supervise_image.ipynb": SUPERVISE,
    "06_collecte_api.ipynb": COLLECTE,
}


def construire(sans_execution: bool) -> None:
    CARNETS.mkdir(exist_ok=True)
    ecrits = []
    for nom, cellules in CARNETS_A_CONSTRUIRE.items():
        carnet = nbformat.v4.new_notebook(cells=cellules)
        carnet.metadata = {
            "kernelspec": {"display_name": "pcc", "language": "python", "name": "pcc"},
            "language_info": {"name": "python", "version": "3.12"},
        }
        chemin = CARNETS / nom
        nbformat.write(carnet, chemin)
        ecrits.append(chemin)
        print(f"  écrit  {chemin.relative_to(ROOT)}  ({len(cellules)} cellules)")

    # L'intégration continue vérifie le formatage des cellules de code au même
    # titre que celui des fichiers Python. Le code des carnets étant écrit à la
    # main dans ce fichier, on le normalise ici plutôt que de laisser la CI
    # échouer sur des retours à la ligne.
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--quiet", *map(str, ecrits)], check=False
    )
    print("  formaté")

    if sans_execution:
        print("\nConstruits sans exécution. `python scripts/run_notebooks.py` pour les exécuter.")
        return

    print()
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_notebooks import executer

    for nom in CARNETS_A_CONSTRUIRE:
        executer(CARNETS / nom)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sec", action="store_true", help="construire sans exécuter")
    construire(p.parse_args().sec)
