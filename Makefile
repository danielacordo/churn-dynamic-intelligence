.PHONY: run simulate test streamlit lint install

install:
	pip install -r requirements.txt

run:
	python main.py

simulate:
	python main.py --simulate

test:
	pytest tests/ -v

lint:
	ruff check .

streamlit:
	streamlit run app.py
