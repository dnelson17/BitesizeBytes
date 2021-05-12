from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

p = Path.cwd()
max_cores = 6
core_list = [2**i for i in range(max_cores)]
func_list = ["MyFunc","Lapack"]
normalise = lambda x, y: x/y

for func_name in func_list:
    total_df = pd.read_pickle(f"{func_name}/Time_dfs/total_df.pkl")
    total_df = total_df.sort_index()
    total_df = total_df.groupby(total_df.index).mean()
    print(total_df)

    scatter_df = pd.read_pickle(f"{func_name}/Time_dfs/scatter_df.pkl")
    scatter_df = scatter_df.sort_index()
    scatter_df = scatter_df.groupby(scatter_df.index).mean()
    norm_scatter_df = scatter_df.combine(total_df, normalise)
    print("norm_scatter_df")
    print(norm_scatter_df)

    calc_df = pd.read_pickle(f"{func_name}/Time_dfs/calc_df.pkl")
    calc_df = calc_df.sort_index()
    calc_df = calc_df.groupby(calc_df.index).mean()
    norm_calc_df = calc_df.combine(total_df, normalise)
    print("norm_calc_df")
    print(norm_calc_df)

    gather_df = pd.read_pickle(f"{func_name}/Time_dfs/gather_df.pkl")
    gather_df = gather_df.sort_index()
    gather_df = gather_df.groupby(gather_df.index).mean()
    norm_gather_df = gather_df.combine(total_df, normalise)
    print("norm_gather_df")
    print(norm_gather_df)


    bar_width = 0.1
    mean = sum(range(6))/len(range(6))
    for i in range(max_cores):
        bar_l = [j-1+1.5*bar_width*(i-mean) for j in range(len(total_df.index))]
        norm_scatter_sub_df = norm_scatter_df[[2**i]].T.values[0]
        norm_calc_sub_df = norm_calc_df[[2**i]].T.values[0]
        norm_gather_sub_df = norm_gather_df[[2**i]].T.values[0]

        scatter_plt =  plt.bar(bar_l,
                            norm_scatter_sub_df,
                            width=bar_width,
                            label="Scatter",
                            alpha=0.5,
                            color="b")

        calc_plt =  plt.bar(bar_l,
                            norm_calc_sub_df,
                            width=bar_width,
                            label="Calc",
                            alpha=0.5,
                            bottom=norm_scatter_sub_df,
                            color="g")

        gather_plt = plt.bar(bar_l,
                            norm_gather_sub_df,
                            width=bar_width,
                            label="Gather",
                            alpha=0.5,
                            bottom=norm_scatter_sub_df+norm_calc_sub_df,
                            color="r")

    plt.xticks([j-1 for j in range(len(total_df.index))], calc_df.index)
    plt.xlabel("Matrix order")
    plt.ylabel("Normalised Runtime")
    plt.legend(["Scatter","Calc","Gather"])
    plt.savefig(f"{p.parent.parent}\Figures\MPI\MPI_{func_name}_barchart_all_cores.png")
    plt.clf()
