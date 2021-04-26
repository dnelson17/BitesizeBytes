from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

pkl_list = ["read", "calc", "write", "total"]
core_list = [2**i for i in range(6)]
p = Path.cwd()

for pkl_name in pkl_list:
    time_df = pd.read_pickle(pkl_name)
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    speedup_df = speedup_df.append( ideal_df )
    speedup_df = speedup_df.T
    speedup_df.plot()
    plt.xlabel("Number of Cores")
    plt.ylabel("Runtime Speedup")
    plt.legend()
    plt.savefig(f"{p.parent.parent}\Figures\MPI_myfunc_{pkl_name}_speedup.png")
