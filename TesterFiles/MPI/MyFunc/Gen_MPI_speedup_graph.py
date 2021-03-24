from matplotlib import pyplot as plt
import pandas as pd

pkl_list = ["scatter_df.pkl", "calc_df.pkl", "gather_df.pkl", "total_df.pkl"]

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
    plt.savefig(f"MPI_myfunc_{pkl_name[:-7]}_speedup.png", dpi=1000)
