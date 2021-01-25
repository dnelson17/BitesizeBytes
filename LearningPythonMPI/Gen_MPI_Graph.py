from matplotlib import pyplot as plt

with open('results.txt') as f:
    lines = [line.rstrip() for line in f]

times = ['']
for i in range(len(lines)):
    if lines[i] != '':
        times[-1] += (str(lines[i]) + ' ')
    else:
        times.append('')

print(times)

for i in range(len(times)):
    speedups.append( times[0] / times[i] )
