import sys
import matplotlib.pyplot as plt
import numpy as np

file = sys.argv[-1]
with open(file) as f:
    cnt = f.readlines()

preline = None
dists = []
for line in cnt:
    if line.endswith('sample...') 
        if preline is not None:
            dists.append(float(line.split(':')[-1].split()[-1]))
        else: idx = int(line.split()[-2])
    preline = line
    if idx = 100: break

ths = np.linspace(0,10, 1000)
amt = []
dists = np.array(sorted(dists))
for i in ths:
    idx = np.argmin(np.abs(dists-i))
    if dists[idx] < i: idx += 1
    if i == 0: amt.append(idx)
    else: amt.append(idx-amt[-1])
    if idx == 99: break

plt.bar(ths[:len(amt)], amt)
plt.savefig('res/{}.png'.format(file[:-4]))
