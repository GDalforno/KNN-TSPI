setup-dev:
	pip install poetry==1.8.0 pre-commit==3.7.0
	pre-commit install --hook-type pre-commit --hook-type pre-push
	pre-commit run --all-files
	poetry install

test-dev:
	pytest -vv -W ignore

setup-ci:
	python3 -m pip install poetry==1.8.0
	python3 -m poetry install

lint-ci:
	python3 -m ruff check knn_tspi/
	python3 -m mypy knn_tspi/ --explicit-package-bases --install-types --non-interactive

test-ci:
	python3 -m pytest --cov -vv -W ignore --cov-fail-under=90 --cov-report term-missing

build-ci:
	python3 -m build
