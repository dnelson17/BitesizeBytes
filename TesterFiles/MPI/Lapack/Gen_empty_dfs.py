import pandas as pd

pkl_list = ["scatter_df.pkl", "calc_df.pkl", "gather_df.pkl", "total_df.pkl"]
core_list = [2**i for i in range(6)]

for pkl_name in pkl_list:
    df = pd.DataFrame(columns=core_list)
    df.to_pickle(pkl_name)
