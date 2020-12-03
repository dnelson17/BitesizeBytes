from matplotlib import pyplot as plt

times = [17.8940,9.1336,4.9853,2.6625,1.3070]

speedups = []

for i in range(len(times)):
    speedup.append( times[0] / times[i] )

no_cores = [1,2,4,8,16]

plt.plot(no_cores,speedup)
plt.plt(no_cores,no_cores)
plt.xlim(1,16)
plt.xlim(1,16)
plt.xlabel("No of cores")
plt.ylabel("Speed-up")
plt.legend()
plt.show()
