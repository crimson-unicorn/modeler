from __future__ import division
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--interval', required=True)
parser.add_argument('-i', '--input', required=True)
args = parser.parse_args()

unique_sketches = set()
sketches = list()
total = 0
with open(args.input, 'r') as f:
    for line in f:
        total += 1
        sketch = line.strip().split()
        if len(sketches) == 0:
            sketches.append(frozenset(sketch))
        else:
            if sketches[len(sketches) - 1] != frozenset(sketch):
                sketches.append(frozenset(sketch))
        unique_sketches.add(frozenset(sketch))

print "File name : " + str(args.input)
print "Interval: " + str(args.interval)
print "Total # of Sketches: " + str(total)
print "# of Unique Sketches: " + str(len(unique_sketches)) + " ; # of Sketches: " + str(len(sketches))
print "% of Unique Sketches: " + str(len(unique_sketches) / total)

