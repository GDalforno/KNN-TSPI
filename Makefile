setup-dev:
	pip install pytest==7.1.2 pytest-cov==4.0.0 pytest-mock==3.10.0 pre-commit==3.2.2 black==23.1.0 mypy==1.4.1
	pip install -r requirements.txt
	pre-commit install --hook-type pre-commit --hook-type pre-push
	pre-commit run --all-files

test-dev:
	pytest -vv -W ignore

setup-ci:
	python3 -m pip install --upgrade pip build twine
	pip install pytest==7.1.2 pytest-cov==4.0.0 pytest-mock==3.10.0 black==23.1.0 mypy==1.4.1
	pip install -r requirements.txt

lint-ci:
	python3 -m black knn_tspi/

test-ci:
	python3 -m pytest --cov -vv -W ignore --cov-fail-under=90 --cov-report term-missing

build-ci:
	python3 -m build
