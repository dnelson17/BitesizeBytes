from matplotlib import pyplot as plt
import pandas as pd

time_df = pd.read_pickle(f"time_dfs/read_df.pkl")

print(time_df.info())

print(time_df.to_string())
time_df = time_df.sort_index()
print(time_df.to_string())
time_df = time_df.groupby(time_df.index).mean()
print(time_df.to_string())
