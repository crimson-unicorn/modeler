# Unicorn Evolutionary Modeler
Unicorn uses an evolutionary modeler to model the benign behavior of the system and detect anomalies.

## Usage
You can either run our Python scripts directly or use our Makefile template that runs them in a virtual environment together (recommended).
To run modeler manually, you must install `numpy`, `scipy`, and `sklearn` first:
```
pip install numpy scipy sklearn
```
You can then run:
```
python model.py -h
```
to see the required and optional arguments.
If you want to use a virtual environment, please use the `Makefile` template and make sure you have `virtualenv` installed.
You can run
```
make example
```
and clean up with:
```
make clean
```

