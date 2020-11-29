toy:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy sklearn ; \
		python model.py -t ../../data/train_toy/ -u ../../data/test_toy/ -m mean -c 0 -v > ../../output/toy-results.txt

example:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy sklearn ; \
		python model.py -h

clean:
	rm -rf venv
