from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

p = Path.cwd()
core_list = [2**i for i in range(6)]

total_df = pd.read_pickle("Time_dfs/total_df.pkl")
total_df = total_df.sort_index()
total_df = total_df.groupby(total_df.index).mean()

print(total_df)

normalise = lambda x, y: x/y

read_df = pd.read_pickle("Time_dfs/read_df.pkl")
read_df = read_df.sort_index()
read_df = read_df.groupby(read_df.index).mean()

norm_read_df = read_df.combine(total_df, normalise)
print("norm_read_df")
print(norm_read_df)

calc_df = pd.read_pickle("Time_dfs/calc_df.pkl")
calc_df = calc_df.sort_index()
calc_df = calc_df.groupby(calc_df.index).mean()

norm_calc_df = calc_df.combine(total_df, normalise)
print("norm_calc_df")
print(norm_calc_df)


write_df = pd.read_pickle("Time_dfs/write_df.pkl")
write_df = write_df.sort_index()
write_df = write_df.groupby(write_df.index).mean()

norm_write_df = write_df.combine(total_df, normalise)
print("norm_write_df")
print(norm_write_df)

#sub_df = pd.concat([read_df,calc_df,write_df],keys=["read","calc","write"],axis=1)


bar_l = [i+1 for i in range(len(total_df.index))]
bar_width = 0.75

read_plt =  plt.bar(bar_l,
                    norm_read_df[[1]].T.values[0],
                    width=bar_width,
                    label="Read",
                    alpha=0.5,
                    color="b")

calc_plt =  plt.bar(bar_l,
                    norm_calc_df[[1]].T.values[0],
                    width=bar_width,
                    label="Calc",
                    alpha=0.5,
                    bottom=norm_read_df[[1]].T.values[0],
                    color="g")

wrtie_plt = plt.bar(bar_l,
                    norm_write_df[[1]].T.values[0],
                    width=bar_width,
                    label="Write",
                    alpha=0.5,
                    bottom=norm_read_df[[1]].T.values[0]+norm_calc_df[[1]].T.values[0],
                    color="r")

plt.xticks(bar_l, calc_df.index)
plt.xlabel("Matrix order")
plt.ylabel("Normalised Runtime")
plt.legend()
plt.xlim([min(bar_l)-bar_width, max(bar_l)+bar_width])
plt.savefig(f"{p.parent.parent}\Figures\MPI_IO\MPI_IO_barchart.png")
