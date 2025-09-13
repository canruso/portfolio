.PHONY: setup run

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
\tstreamlit run app/app.py
