from matplotlib import pyplot as plt

with open('results.txt') as f:
    lines = [line.rstrip() for line in f]

times = ['']
for i in range(len(lines)):
    if lines[i] != '':
        times[-1] += (str(lines[i]) + ' ')
    else:
        times.append('')
        
times = times[1:-1]

for time in times:
    temp = time.split(" ")
    temp = temp[:-1]
    temp[0] = int(temp[0])
    temp[1] = int(temp[1])-1
    for j in range(2,len(temp)):
        temp[j] = float(temp[j])
    print(temp)


speedups = []

#for i in range(len(times)):
#    speedups.append( times[0] / times[i] )
