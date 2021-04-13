from matplotlib import pyplot as plt
import pandas as pd

pkl_list = ["read", "calc", "write", "total"]
core_list = [2**i for i in range(6)]

for pkl_name in pkl_list:
    time_df = pd.read_pickle(f"time_dfs/{pkl_name}_df.pkl")
    print(f"\n\n{pkl_name}")
    print(time_df.to_string())
    time_df = time_df.sort_index()
    print(time_df.to_string())
    time_df = time_df.groupby(time_df.index).mean()
    print(time_df.to_string())
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    speedup_df = speedup_df.append( ideal_df )
    print(speedup_df.to_string())
    speedup_df = speedup_df.T
    speedup_df.plot()
    plt.xlabel("Number of Cores")
    plt.ylabel("Runtime Speedup")
    plt.legend()
    #plt.savefig(f"C:\\University\\Project\\BitesizeBytes\\TesterFiles\\Figures\\MPI_lapack_{pkl_name[:-7]}_speedup.jpeg", dpi=1000)
    plt.savefig(f"MPI_IO_{pkl_name}_speedup.png")
