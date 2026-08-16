"""Collecte de produits d'épicerie fine via une API publique.

Linda demande de tester l'ouverture du catalogue à l'épicerie fine en
récupérant les dix premiers produits associés au champagne, avec cinq champs
précis : `foodId`, `label`, `category`, `foodContentsLabel` et `image`.

Ces noms de champs viennent du schéma d'Edamam. Nous interrogeons Open Food
Facts, qui ne demande aucune inscription — le script reste donc exécutable
par n'importe qui, sans clé à transmettre. Cela impose en revanche de faire
correspondre les deux vocabulaires, et cette correspondance est le seul
endroit du script où un jugement intervient : elle est explicitée dans
`CORRESPONDANCE`.

    python collecte_api.py                  # dix produits « champagne »
    python collecte_api.py --terme "foie gras" --nombre 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"

# Point d'entrée v2, documenté et maintenu. L'ancien `cgi/search.pl` fonctionne
# encore mais il est déprécié, et il nous a renvoyé un 503 au premier essai.
URL = "https://world.openfoodfacts.org/api/v2/search"
TENTATIVES = 3

# Open Food Facts demande une identification explicite des clients.
AGENT = "product-category-classifier/1.0 (contact: richard.hugou@gmail.com)"

# Champ demandé  ->  champ Open Food Facts qui lui correspond.
#
# `foodContentsLabel` désigne chez Edamam la liste des composants d'un produit.
# `ingredients_text` est son équivalent le plus direct ici. Les autres
# correspondances sont sans ambiguïté.
CORRESPONDANCE = {
    "foodId": "code",
    "label": "product_name",
    "category": "categories",
    "foodContentsLabel": "ingredients_text",
    "image": "image_url",
}


def interroger(terme: str, nombre: int) -> list[dict]:
    """Les `nombre` premiers produits de la catégorie demandée.

    Le filtre porte sur la catégorie et non sur le texte libre. Une recherche
    plein texte sur « champagne » remonte aussi les produits qui se contentent
    de mentionner le mot — vinaigres, sauces, arômes —, ce qui n'est pas ce
    qu'on cherche à collecter.

    Le service répond parfois 503 sans raison durable ; on réessaie plutôt que
    d'échouer sur un incident passager.
    """
    parametres = urllib.parse.urlencode(
        {
            "categories_tags_en": terme,
            "page_size": nombre,
            "fields": ",".join(CORRESPONDANCE.values()),
        }
    )
    requete = urllib.request.Request(f"{URL}?{parametres}", headers={"User-Agent": AGENT})

    for essai in range(1, TENTATIVES + 1):
        try:
            with urllib.request.urlopen(requete, timeout=60) as reponse:  # noqa: S310 — URL fixe
                return json.load(reponse).get("products", [])[:nombre]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as erreur:
            if essai == TENTATIVES:
                raise
            attente = 2**essai
            print(f"  tentative {essai} — {erreur}, nouvelle tentative dans {attente} s")
            time.sleep(attente)
    return []


def normaliser(produit: dict) -> dict:
    """Un produit Open Food Facts, ramené aux cinq champs demandés.

    Les champs absents deviennent une chaîne vide plutôt que d'être omis :
    le fichier doit avoir la même forme pour tous les produits, et l'absence
    est elle-même une information à faire remonter.
    """
    return {
        demande: str(produit.get(source, "") or "").strip()
        for demande, source in CORRESPONDANCE.items()
    }


def main(terme: str, nombre: int, sortie: Path) -> None:
    print(f"Recherche « {terme} » sur Open Food Facts")
    produits = [normaliser(p) for p in interroger(terme, nombre)]
    if not produits:
        print("Aucun produit renvoyé.", file=sys.stderr)
        raise SystemExit(1)

    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("w", encoding="utf-8", newline="") as fichier:
        redacteur = csv.DictWriter(fichier, fieldnames=list(CORRESPONDANCE))
        redacteur.writeheader()
        redacteur.writerows(produits)

    print(f"  {len(produits)} produits écrits dans {sortie}\n")
    for champ in CORRESPONDANCE:
        remplis = sum(1 for p in produits if p[champ])
        print(f"  {champ:20s} renseigné pour {remplis}/{len(produits)} produits")

    print("\nTrois premiers produits")
    for p in produits[:3]:
        print(f"  {p['foodId']:>16}  {p['label'][:52] or '(sans nom)'}")
        print(f"                    catégories : {p['category'][:70] or '(vide)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terme", default="champagne", help="terme de recherche")
    p.add_argument("--nombre", type=int, default=10, help="nombre de produits")
    p.add_argument("--sortie", type=Path, default=REPORTS / "produits_champagne.csv")
    a = p.parse_args()
    main(a.terme, a.nombre, a.sortie)
