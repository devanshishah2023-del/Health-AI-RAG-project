.PHONY: install run test demo lint clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

# Run the test suite against the offline TF-IDF backend (no downloads/network).
test:
	EMBEDDING_BACKEND=tfidf pytest -q

# Generate the 5 required sample outputs into the README-friendly format.
demo:
	python scripts/demo.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
	rm -f logs/research_log.jsonl logs/research_log.db
