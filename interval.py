import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True)
args = parser.parse_args()

unique_sketches = set()
sketches = list()
with open(args.input, 'r') as f:
    for line in f:
        sketch = line.strip().split()
        if len(sketches) == 0:
            sketches.append(frozenset(sketch))
        else:
            if sketches[len(sketches) - 1] != frozenset(sketch):
                sketches.append(frozenset(sketch))
        unique_sketches.add(frozenset(sketch))

print "# of Unique Sketches: " + len(unique_sketches) + " ; # of Sketches: " + len(sketches)