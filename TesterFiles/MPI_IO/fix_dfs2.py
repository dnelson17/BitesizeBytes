from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

core_list = [2**i for i in range(6)]

df = pd.DataFrame(columns=core_list)
print(df)
print(df.info())
print("\n")

read_times = [21.0,13.0,7.0,5.0,3.0,2.0]
max_cores = 32
mat_size = 1024

for i in range(6):
    read_time = read_times[i]
    size = core_list[i]
    size_power = int(np.log2(size))
    if size == 1:
        df = df.append( pd.DataFrame([tuple([read_time if i==0 else 0.0 for i in range(int(np.log2(max_cores))+1)])],columns=core_list, index=[mat_size]) )
    elif size > 1:
        df.iloc[-1, size_power] = read_time
print(df)
print(df.info())
print("\n")
