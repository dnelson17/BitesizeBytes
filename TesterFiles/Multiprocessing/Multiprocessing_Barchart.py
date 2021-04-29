from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

p = Path.cwd()
core_list = [2**i for i in range(6)]
func_list = ["MyFunc","Lapack"]

for func_name in func_list:
    total_df = pd.read_pickle(f"{func_name}/Time_dfs/total_df.pkl")
    total_df = total_df.sort_index()
    total_df = total_df.groupby(total_df.index).mean()

    print(total_df)

    normalise = lambda x, y: x/y

    scatter_df = pd.read_pickle(f"{func_name}/Time_dfs/scatter_df.pkl")
    scatter_df = scatter_df.sort_index()
    scatter_df = scatter_df.groupby(scatter_df.index).mean()
    norm_scatter_df = scatter_df.combine(total_df, normalise)
    norm_scatter_df = norm_scatter_df[[1]].T.values[0]
    print("norm_scatter_df")
    print(norm_scatter_df)

    calc_df = pd.read_pickle(f"{func_name}/Time_dfs/calc_df.pkl")
    calc_df = calc_df.sort_index()
    calc_df = calc_df.groupby(calc_df.index).mean()
    norm_calc_df = calc_df.combine(total_df, normalise)
    norm_calc_df = norm_calc_df[[1]].T.values[0]
    print("norm_calc_df")
    print(norm_calc_df)

    gather_df = pd.read_pickle(f"{func_name}/Time_dfs/gather_df.pkl")
    gather_df = gather_df.sort_index()
    gather_df = gather_df.groupby(gather_df.index).mean()
    norm_gather_df = gather_df.combine(total_df, normalise)
    norm_gather_df = norm_gather_df[[1]].T.values[0]
    print("norm_gather_df")
    print(norm_gather_df)


    bar_l = [i+1 for i in range(len(total_df.index))]
    bar_width = 0.65

    scatter_plt =  plt.bar(bar_l,
                        norm_scatter_df,
                        width=bar_width,
                        label="Scatter",
                        alpha=0.5,
                        color="b")

    calc_plt =  plt.bar(bar_l,
                        norm_calc_df,
                        width=bar_width,
                        label="Calc",
                        alpha=0.5,
                        bottom=norm_scatter_df,
                        color="g")

    gather_plt = plt.bar(bar_l,
                        norm_gather_df,
                        width=bar_width,
                        label="Gather",
                        alpha=0.5,
                        bottom=norm_scatter_df+norm_calc_df,
                        color="r")

    plt.xticks(bar_l, calc_df.index)
    plt.xlabel("Matrix order")
    plt.ylabel("Normalised Runtime")
    plt.legend()
    plt.xlim([min(bar_l)-bar_width, max(bar_l)+bar_width])
    plt.savefig(f"{p.parent.parent}\Figures\Multiprocessing\Multiprocessing_{func_name}_barchart.png")
    plt.clf()
