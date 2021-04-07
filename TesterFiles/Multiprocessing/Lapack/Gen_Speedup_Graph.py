from matplotlib import pyplot as plt
import pandas as pd

core_list = [2**j for j in range(6)]

pkl_list = ["calc","total"]

for pkl_name in pkl_list:
    print(pkl_name)
    time_df = pd.read_pickle(f"TimeResults/{pkl_name}_time_df_lapack.pkl")
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    print(time_df.to_string())
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    speedup_df = speedup_df.append( ideal_df )
    speedup_df = speedup_df.T
    print(speedup_df)
    speedup_df.plot()
    plt.xlabel("Number of Cores")
    plt.ylabel("Runtime Speedup")
    plt.legend()
    #plt.show()
    plt.savefig(f"multiprocessing_{pkl_name}_speedup_lapack.png")
