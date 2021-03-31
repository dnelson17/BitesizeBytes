import pandas as pd

time_df = pd.read_pickle(f"time_df_libraries.pkl")

time_df = time_df.sort_index()
time_df = time_df.groupby(time_df.index).mean()

print(time_df.to_string())
