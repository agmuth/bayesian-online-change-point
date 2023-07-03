format:
	poetry run black bocd/.
	poetry run isort bocd/.
	
lint: 
	poetry run ruff check bocd/. --fix

test:
	poetry run pytest tests/.