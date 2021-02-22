from matplotlib import pyplot as plt

with open('mpi_io_results.txt') as f:
    lines = [line.rstrip() for line in f]

times = ['']
for i in range(len(lines)):
    if lines[i] != '':
        times[-1] += (str(lines[i]) + ' ')
    else:
        times.append('')

times = times[:-1]

#print(times)
"""
data=[]
for time in times:
    temp = time.split(" ")
    temp = temp[:-1]
    new_time = []
    new_time.append(int(temp[0]))
    new_time.append(int(temp[1])-1)
    sum = 0
    for j in range(2,len(temp)):
        temp[j] = float(temp[j])
    new_time.append(max(temp[2:]))
    data.append(new_time)

final_times = [[]]

current_mat_size = data[0][0]
for i in range(0,len(data)):
    if data[i][0] == current_mat_size:
        final_times[-1].append(data[i][2])
    else:
        final_times.append([])
        current_mat_size = data[i][0]
        final_times[-1].append(data[i][2])

#print(final_times)

speedup_mat = []
"""

#------------
final_times_1 = []
final_times_2 = []
for time in times:
    temp = time.split(" ")
    temp = temp[:-1]
    for j in range(len(temp)):
        temp[j] = float(temp[j])
    print(temp)
    time_temp_1 = [temp[2],temp[5:7],temp[10:14],temp[19:27]]
    time_temp_2 = [temp[3],temp[7:9],temp[14:18],temp[27:35]]
    #print(time_temp_1)
    print(time_temp_2)
    avg_time_1 = [temp[2],sum(temp[5:7])/2,sum(temp[10:14])/4,sum(temp[19:27])/8]
    avg_time_2 = [temp[3],sum(temp[7:9])/2,sum(temp[14:18])/4,sum(temp[27:35])/8]
    print(avg_time_2)
    final_times_1.append(avg_time_1)
    final_times_2.append(avg_time_2)

#------------
speedup_mat = []
for i in range(len(final_times_2)):
  speedup_mat.append([])
  for j in range(len(final_times_2[i])):
    speedup_mat[i].append( final_times_2[i][0] / final_times_2[i][j] )

print("\n\n\nSpeedup mat:")
print(speedup_mat)

no_cores = [1,2,4,8]

for x in range(len(speedup_mat)):
  plt.plot(no_cores,speedup_mat[x], label = f"Actual - {2**(x+3)}")
plt.plot(no_cores,no_cores, label = "Ideal")
#plt.xlim(1,8)
#plt.ylim(1,8)
plt.xlabel("Number of cores")
plt.ylabel("Runtime speed-up")
plt.legend()
plt.show()
