from matplotlib import pyplot as plt
import pandas as pd

time_df = pd.read_pickle(f"time_dfs/read_df.pkl")

print(time_df)

print(time_df.iloc[0])
