from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import re

def read_df_pickle(pkl_name):
    time_df = pd.read_pickle(pkl_name)
    #time_df = time_df.sort_index()
    #time_df = time_df.groupby(time_df.index).mean()
    return time_df


def calc_FLOPS(df):
    flops_df = df.apply(lambda x: ((10**x.name)*7)/(x*(10**6)), axis=1, result_type='expand')
    return flops_df


def print_overleaf_table(df):
    df = df.round(decimals=2)
    #df = df.sort_index()
    #df = df.groupby(df.index).mean()
    df_string = df.to_string().replace("\n","\\\_\n")
    df_string = re.sub(' +', ' ', df_string)
    df_string = df_string.replace(" ","&")
    df_string = df_string.replace("_"," ")
    df_string+="\\\\"
    print(df_string)


def main():
    max_cores = 6
    core_list = [2**i for i in range(max_cores)]
    p = Path.cwd()
    
    strong_time_df = read_df_pickle(f"Time_dfs/time_df_strong_scaling.pkl")
    flops_df = calc_FLOPS(strong_time_df)
    flops_df.index = [f"$10^{{{m}}}$" for m in range(4,11)]
    print_overleaf_table(flops_df)


if __name__ == '__main__':
    main()


