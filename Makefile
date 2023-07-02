format:
	python run black .
	python run isort .
	
lint: 
	python run ruff check . --fix

test:
	python run pytest tests/.