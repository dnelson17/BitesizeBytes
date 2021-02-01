from matplotlib import pyplot as plt

#Open file
with open('scatter_results.txt') as f:
    lines = [line.rstrip() for line in f]

#Read in each line
times = ['']
for i in range(len(lines)):
    if lines[i] != '':
        times[-1] += (str(lines[i]) + ' ')
    else:
        times.append('')

#Take average time for each instance
data=[]
for time in times:
    temp = time.split(" ")
    temp = temp[:-1]
    new_time = []
    new_time.append(int(temp[0]))
    new_time.append(int(temp[1]))
    for j in range(2,len(temp)):
        temp[j] = float(temp[j])
    new_time.append((sum(temp[2:])/len(temp[2:])))
    data.append(new_time)

final_times = [[]]

#Group all results corresponding to matching matrix sizes
current_mat_size = data[0][0]
for i in range(0,len(data)):
    if data[i][0] == current_mat_size:
        final_times[-1].append(data[i][2])
    else:
        final_times.append([])
        current_mat_size = data[i][0]
        final_times[-1].append(data[i][2])

speedup_mat = []

#Turn raw times into speed-ups
for i in range(len(final_times)):
  speedup_mat.append([])
  for j in range(len(final_times[i])):
    speedup_mat[i].append( final_times[i][0] / final_times[i][j] )

#Core list for x-axis
no_cores = [1,2,4,8]

for x in range(len(speedup_mat)):
  plt.plot(no_cores,speedup_mat[x], label = f"Actual - {int(data[4*x][0])}")
plt.plot(no_cores,no_cores, label = "Ideal")
plt.xlim(1,8)
plt.ylim(1,8)
plt.xlabel("Number of cores")
plt.ylabel("Runtime speed-up")
plt.legend()
plt.show()
