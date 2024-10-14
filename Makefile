.PHONY: help
help:
	@echo "Available targets:"
	@echo "  lint       - Run Black and Isort linters"
	@echo "  mypy       - Run MyPy"
	@echo "  flake8     - Run Flake8"
	@echo "  check      - Run all checks (Black, Isort, MyPy, Flake8)"

lint:		## Run Black and Isort linters
	@black .
	@isort .

mypy:		## Run MyPy
	@mypy .

flake8:		## Run Flake8
	@flake8 .

check:    ## Run all checks for Black, Isort, MyPy, Flake8
	- @echo "Black"
	- @black --check .
	- @echo "Isort"
	- @isort --check .
	- @echo "MyPY"
	- @mypy .
	- @echo "Flake8"
	- @flake8 .