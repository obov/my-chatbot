venv:
	. venv/bin/activate

install:
	pip3 install -r requirements.txt || pip install -r requirements.txt

.PHONY: install venv