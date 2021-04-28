import pandas as pd

pkl_list = ["read", "calc", "write", "total"]
core_list = [2**i for i in range(6)]

for pkl_name in pkl_list:
    df = pd.DataFrame(columns=core_list)
    df.to_pickle(f"Time_dfs/{pkl_name}_df.pkl")
