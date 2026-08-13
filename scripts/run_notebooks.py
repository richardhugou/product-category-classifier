"""Exécute les carnets en place, sorties conservées.

Un carnet livré sans ses sorties oblige le lecteur à l'exécuter pour savoir ce
qu'il produit. On les rejoue donc et on les réécrit avec leurs résultats.

    python scripts/run_notebooks.py             # tous les carnets
    python scripts/run_notebooks.py 04 05       # ceux dont le nom commence ainsi
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
CARNETS = ROOT / "notebooks"
DELAI = 1800


def executer(chemin: Path) -> bool:
    """Rejoue un carnet et le réécrit. Renvoie False en cas d'échec."""
    carnet = nbformat.read(chemin, as_version=4)
    # Le répertoire d'exécution est celui du carnet : les chemins relatifs
    # qu'il contient (`sys.path.insert(0, "..")`, `../reports/`) valent aussi
    # bien pour Jupyter que pour cette exécution automatique.
    client = NotebookClient(
        carnet, timeout=DELAI, kernel_name="pcc", resources={"metadata": {"path": str(CARNETS)}}
    )
    depart = time.perf_counter()
    try:
        client.execute()
    except Exception as erreur:
        print(f"  ÉCHEC  {chemin.name} — {type(erreur).__name__}: {str(erreur)[:160]}")
        return False
    nbformat.write(carnet, chemin)
    print(f"  ok     {chemin.name}  ({time.perf_counter() - depart:.0f} s)")
    return True


def main(prefixes: list[str]) -> None:
    carnets = sorted(CARNETS.glob("*.ipynb"))
    if prefixes:
        carnets = [c for c in carnets if any(c.name.startswith(p) for p in prefixes)]
    if not carnets:
        print("Aucun carnet ne correspond.", file=sys.stderr)
        raise SystemExit(1)

    echecs = [c for c in carnets if not executer(c)]
    if echecs:
        print(f"\n{len(echecs)} carnet(s) en échec.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
