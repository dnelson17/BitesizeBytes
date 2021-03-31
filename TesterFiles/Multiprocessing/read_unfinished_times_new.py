import pandas as pd

program_type = "lapack"
#program_type = "myfunc"

core_list = [2**j for j in range(6)]

pkl_list = ["Lapack/send_time_df_lapack.pkl","Lapack/calc_time_df_lapack.pkl","Lapack/recv_time_df_lapack.pkl","Lapack/total_time_df_lapack.pkl"]

for pkl_name in pkl_list:
    print(pkl_name)
    time_df = pd.read_pickle(pkl_name)
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    print(time_df.to_string())

"""
time_df = time_df.sort_index()
time_df = time_df.groupby(time_df.index).mean()

speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])

speedup_df = speedup_df.append( ideal_df )

print(speedup_df)

time_df.to_pickle(f"time_df_{program_type}.pkl")
speedup_df.to_pickle(f"speedup_df_{program_type}.pkl")
"""
