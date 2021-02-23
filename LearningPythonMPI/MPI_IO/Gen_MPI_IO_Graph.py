from matplotlib import pyplot as plt
import numpy as np

with open('mpi_io_results.txt') as f:
    lines = [line.rstrip() for line in f]

times = ['']
for i in range(len(lines)):
    if lines[i] != '':
        times[-1] += (str(lines[i]) + ' ')
    else:
        times.append('')

max_cores = 8
n = int(np.log2(max_cores)+1)

iters = 10

final_times_calc = []
final_times_total = []

for time in times:
    temp = time.split(" ")
    temp = temp[:-1]
    print(temp)
    new_time_calc = []
    new_time_total = []
    for i in range(n):
        time_sum_calc = 0
        time_sum_total = 0
        for j in range(iters):
            time_sum_calc += float(temp[2+3*i+3*n*j])
            time_sum_total += float(temp[3+3*i+3*n*j])
        new_time_calc.append(time_sum_calc/iters)
        new_time_total.append(time_sum_total/iters)
    final_times_calc.append(new_time_calc)
    final_times_total.append(new_time_total)

assert len(final_times_calc) == len(final_times_total)

speedup_mat_calc = []
speedup_mat_total = []
for i in range(len(final_times_calc)):
    speedup_mat_calc.append([])
    speedup_mat_total.append([])
    for j in range(len(final_times_calc[i])):
        speedup_mat_calc[i].append( final_times_calc[i][0] / final_times_calc[i][j] )
        speedup_mat_total[i].append( final_times_total[i][0] / final_times_total[i][j] )

#print("\n\n\nSpeedup mat:")
#print(speedup_mat_calc)

no_cores = [2**i for i in range(n)]

for x in range(len(speedup_mat_calc)):
    plt.plot(no_cores,speedup_mat_total[x], label = f"Actual - {2**(x+7)}")
plt.plot(no_cores,no_cores, label = "Ideal")
#plt.xlim(1,8)
#plt.ylim(1,8)
plt.xlabel("Number of cores")
plt.ylabel("Runtime speed-up")
plt.legend()
plt.show()
