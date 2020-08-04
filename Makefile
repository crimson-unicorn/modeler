example:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy sklearn ; \
		python model.py -h

clean:
	rm -rf venv
