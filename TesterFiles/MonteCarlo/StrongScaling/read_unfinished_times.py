import pandas as pd

core_list = [2**j for j in range(6)]

time_df = pd.read_pickle("time_df_strong_scaling.pkl")
print(time_df)

time_df = time_df.sort_index()
time_df = time_df.groupby(time_df.index).mean()

speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])

speedup_df = speedup_df.append( ideal_df )

print(speedup_df)

time_df.to_pickle("time_df_strong_scaling.pkl")
speedup_df.to_pickle("speedup_df_strong_scaling.pkl")
