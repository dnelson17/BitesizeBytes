import pandas as pd
import re

time_df = pd.read_pickle(f"time_df_libraries.pkl")

print(time_df)

time_df = time_df.sort_index()
time_df = time_df.groupby(time_df.index).mean()

df_string = time_df.to_string().replace("\n","\\\_\n")
df_string = re.sub(' +', ' ', df_string)
df_string = df_string.replace(" ","&")
df_string = df_string.replace("_"," ")

print(df_string)
