.PHONY: help all benchmark optimize figures demo test lint notebooks clean

help:                        ## affiche cette aide
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

all: benchmark figures       ## reproduit tous les artefacts de reports/

benchmark:                   ## entraîne et évalue (ENCODERS=1 IMAGES=1 pour tout)
	python benchmark.py $(if $(ENCODERS),--encoders,) $(if $(IMAGES),--images,)

optimize:                    ## explore les hyperparamètres (HOLDOUT=1 pour la méthode naïve)
	python optimize.py --images $(if $(HOLDOUT),--holdout,)

figures:                     ## regénère les quatre figures
	python -m src.figures

demo:                        ## comparaison texte / image / fusion (travail complémentaire)
	streamlit run app.py

demo-mission:                ## le modèle de la partie 4, seul — celui de la soutenance
	streamlit run demo.py

modele:                      ## sérialise le modèle de la partie 4 et vérifie qu'il reproduit 0,867
	python scripts/exporter_modele.py

test:                        ## tests unitaires
	pytest

lint:                        ## lint et format
	ruff check . && ruff format --check .

notebooks:                   ## rejoue les carnets en place (NB=04 pour n'en cibler qu'un)
	python scripts/run_notebooks.py $(NB)

clean:
	rm -rf models reports/*.png reports/*.csv reports/*.json .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
