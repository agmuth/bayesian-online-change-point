format:
	poetry run black bocd/.
	poetry run black tests/.
	poetry run isort bocd/.
	poetry run isort tests/.
	
lint: 
	poetry run ruff check bocd/. --fix
	poetry run ruff check tests/. --fix

test:
	poetry run pytest tests/.