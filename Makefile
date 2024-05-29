setup-dev:
	pip install poetry==1.8.0 pre-commit==3.7.0
	pre-commit install --hook-type pre-commit --hook-type pre-push
	pre-commit run --all-files
	poetry install

test-dev:
	pytest -vv -W ignore
