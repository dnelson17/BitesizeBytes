from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

p = Path.cwd()
max_cores = 6
core_list = [2**i for i in range(max_cores)]
func_list = ["MyFunc","Lapack"]
normalise = lambda x, y: x/y

total_df = pd.read_pickle("Time_dfs/total_df.pkl")
total_df = total_df.sort_index()
total_df = total_df.groupby(total_df.index).mean()
#print(total_df)

read_df = pd.read_pickle("Time_dfs/read_df.pkl")
read_df = read_df.sort_index()
read_df = read_df.groupby(read_df.index).mean()
norm_read_df = read_df.combine(total_df, normalise)
#print("norm_read_df")
#print(norm_read_df)

calc_df = pd.read_pickle("Time_dfs/calc_df.pkl")
calc_df = calc_df.sort_index()
calc_df = calc_df.groupby(calc_df.index).mean()
norm_calc_df = calc_df.combine(total_df, normalise)
#print("norm_calc_df")
#print(norm_calc_df)

write_df = pd.read_pickle("Time_dfs/write_df.pkl")
write_df = write_df.sort_index()
write_df = write_df.groupby(write_df.index).mean()
norm_write_df = write_df.combine(total_df, normalise)
#print("norm_write_df")
#print(norm_write_df)


bar_width = 0.1
mean = sum(range(6))/len(range(6))
for i in range(max_cores):
    bar_l = [j-1+1.5*bar_width*(i-mean) for j in range(len(total_df.index))]
    norm_read_sub_df = norm_read_df[[2**i]].T.values[0]
    norm_calc_sub_df = norm_calc_df[[2**i]].T.values[0]
    norm_write_sub_df = norm_write_df[[2**i]].T.values[0]

    read_plt =  plt.bar(bar_l,
                        norm_read_sub_df,
                        width=bar_width,
                        label="Read",
                        alpha=0.5,
                        color="b")

    calc_plt =  plt.bar(bar_l,
                        norm_calc_sub_df,
                        width=bar_width,
                        label="Calc",
                        alpha=0.5,
                        bottom=norm_read_sub_df,
                        color="g")

    write_plt = plt.bar(bar_l,
                        norm_write_sub_df,
                        width=bar_width,
                        label="Write",
                        alpha=0.5,
                        bottom=norm_read_sub_df+norm_calc_sub_df,
                        color="r")

plt.xticks([j-1 for j in range(len(total_df.index))], calc_df.index)
plt.xlabel("Matrix order")
plt.ylabel("Normalised Runtime")
plt.legend(["Read","Calc","Write"])
plt.show()
#plt.savefig(f"{p.parent.parent}\Figures\MPI_IO\MPI_IO_barchart_all_cores.png")
