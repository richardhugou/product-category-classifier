"""Convertit le rapport en PDF, figures comprises.

Le rapport est écrit en Markdown pour rester lisible et versionnable. Le dépôt
attendu par la plateforme demande un PDF. Ce script fait la conversion en deux
temps : Markdown vers HTML mis en forme, puis impression par le navigateur
déjà installé sur la machine, ce qui évite d'ajouter une chaîne LaTeX complète
pour un seul document.

    python scripts/rapport_pdf.py
    python scripts/rapport_pdf.py --sortie /chemin/rapport.pdf
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "RAPPORT.md"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

STYLE = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body {
  font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1f2328; max-width: none;
}
h1 {
  font-size: 19pt; color: #1a4d80; margin: 2.4em 0 0.8em;
  padding-bottom: 0.3em; border-bottom: 2px solid #4c78a8; page-break-after: avoid;
}
h1:first-of-type { page-break-before: avoid; margin-top: 0; }
h2 { font-size: 13.5pt; color: #24455f; margin: 1.9em 0 0.6em; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #24455f; margin: 1.5em 0 0.5em; page-break-after: avoid; }
p { margin: 0 0 0.85em; text-align: justify; }
strong { color: #12263a; }
em { color: #3d4b57; }
table {
  border-collapse: collapse; width: 100%; margin: 1.2em 0;
  font-size: 9.3pt; page-break-inside: avoid;
}
th {
  background: #eef3f8; color: #1a4d80; text-align: left;
  padding: 6px 9px; border-bottom: 2px solid #4c78a8;
}
td { padding: 5px 9px; border-bottom: 1px solid #e3e8ee; }
tr:last-child td { border-bottom: none; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
  background: #f4f6f8; padding: 1px 4px; border-radius: 3px; color: #8a4b2a;
}
pre {
  background: #f7f9fb; border-left: 3px solid #4c78a8; padding: 10px 13px;
  overflow-x: auto; font-size: 8.8pt; line-height: 1.45; page-break-inside: avoid;
}
pre code { background: none; padding: 0; color: #1f2328; }
img {
  max-width: 78%; height: auto; display: block; margin: 1.1em auto;
  page-break-inside: avoid;
}
hr { border: none; border-top: 1px solid #dde3ea; margin: 2.2em 0; }
blockquote {
  border-left: 3px solid #c9d6e2; margin: 1em 0; padding: 0.2em 0 0.2em 1em; color: #46535f;
}
.couverture { text-align: center; padding: 4em 0 3em; page-break-after: always; }
.couverture h1 {
  border: none; page-break-before: avoid; font-size: 25pt; margin-bottom: 0.4em;
}
.couverture .sous { font-size: 13pt; color: #46535f; margin-bottom: 3em; }
.couverture .auteur { font-size: 11pt; color: #1f2328; }
.couverture .liens { margin-top: 4em; font-size: 10pt; color: #46535f; line-height: 2; }
"""


def _incorporer_images(html: str, base: Path) -> str:
    """Remplace chaque figure par ses données, pour un PDF autonome."""

    def remplacer(correspondance: re.Match) -> str:
        chemin = (base / correspondance.group(1)).resolve()
        if not chemin.exists():
            print(f"  figure introuvable : {chemin}", file=sys.stderr)
            return correspondance.group(0)
        donnees = base64.b64encode(chemin.read_bytes()).decode()
        return f'src="data:image/png;base64,{donnees}"'

    return re.sub(r'src="([^"]+)"', remplacer, html)


def construire_html(source: Path) -> str:
    texte = source.read_text(encoding="utf-8")

    # La page de garde est composée à part : le Markdown ne permet pas de la
    # mettre en forme, et elle porte les liens que le jury doit pouvoir suivre.
    lignes = texte.split("\n")
    fin_entete = next(i for i, ligne in enumerate(lignes) if ligne.startswith("# 1."))
    corps = "\n".join(lignes[fin_entete:])

    contenu = markdown.markdown(
        corps, extensions=["tables", "fenced_code", "attr_list", "md_in_html"]
    )
    contenu = _incorporer_images(contenu, source.parent)

    couverture = """
<div class="couverture">
  <h1>Étude de faisabilité d'un moteur de classification automatique d'articles</h1>
  <div class="sous">À partir des descriptions textuelles et des photographies de produits</div>
  <div class="auteur"><strong>Richard Hugou</strong> — Data Scientist junior</div>
  <div class="auteur">août 2026</div>
  <div class="liens">
    Projet technique :
    <a href="https://github.com/richardhugou/product-category-classifier">github.com/richardhugou/product-category-classifier</a><br>
    Portfolio : <a href="https://portfolio.richardh.fr">portfolio.richardh.fr</a>
  </div>
</div>
"""
    return (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        f"<title>Rapport — Richard Hugou</title><style>{STYLE}</style></head>"
        f"<body>{couverture}{contenu}</body></html>"
    )


def imprimer(html: str, sortie: Path) -> None:
    if not CHROME.exists():
        print(f"Navigateur introuvable : {CHROME}", file=sys.stderr)
        raise SystemExit(1)

    sortie.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as dossier:
        page = Path(dossier) / "rapport.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                str(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={sortie}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )


def main(sortie: Path) -> None:
    html = construire_html(SOURCE)
    imprimer(html, sortie)
    taille = sortie.stat().st_size / 1e6
    print(f"  {sortie}  ({taille:.1f} Mo)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sortie", type=Path, default=ROOT / "docs" / "RAPPORT.pdf")
    main(p.parse_args().sortie)
