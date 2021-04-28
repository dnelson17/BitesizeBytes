from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

def read_df_pickle(pkl_name):
    time_df = pd.read_pickle(pkl_name)
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    return time_df


def apply_norm(time_df,norm_index,core_list):
    norm_df = time_df.apply(lambda x: x/x.iloc[norm_index], axis=1, result_type='expand')
    ideal_df = pd.DataFrame([[1 for _ in range(len(core_list))]],columns=core_list,index=["Ideal"])
    norm_df = norm_df.append( ideal_df )
    return norm_df


def apply_speedup(time_df,core_list):
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    speedup_df = speedup_df.append( ideal_df )
    return speedup_df


def gen_plot(df,p,speedup,pkl_name,func_name):
    df = df.T
    df.plot()
    plt.xlabel("Matrix Order (n)")
    if speedup:
        plt.ylabel("Runtime Speedup")
    else:
        plt.ylabel("Normalised Runtime")
    plt.legend()
    p = Path.cwd()
    if speedup:
        plt.savefig(f"{p.parent.parent}\Figures\Multiprocessing\multiprocessing_{func_name}_{pkl_name}_speedup.png")
    else:
        plt.savefig(f"{p.parent.parent}\Figures\Multiprocessing\multiprocessing_{func_name}_{pkl_name}_norm.png")


def main():
    max_cores = 6
    core_list = [2**i for i in range(max_cores)]
    p = Path.cwd()
    func_list = ["MyFunc","Lapack"]
    speedup_pkl_list = ["calc", "total"]
    norm_pkl_list = ["scatter", "gather"]
    norm_index = 0
    for func_name in func_list:
        for speedup_pkl_name in speedup_pkl_list:
            print(f"\n\n{speedup_pkl_name}")
            time_df = read_df_pickle(f"{func_name}/Time_dfs/{speedup_pkl_name}_df.pkl")
            print(f"time df: \n{time_df}")
            speedup_df = apply_speedup(time_df,core_list)
            print(f"speedup df: \n{speedup_df}")
            gen_plot(speedup_df,p,True,speedup_pkl_name,func_name)
        for norm_pkl_name in norm_pkl_list:
            print(f"\n\n{norm_pkl_name}")
            time_df = read_df_pickle(f"{func_name}/Time_dfs/{norm_pkl_name}_df.pkl")
            if func_name == "Lapack":
                time_df = time_df.drop([256])
            print(f"time df: \n{time_df}")
            norm_df = apply_norm(time_df,0,core_list)
            print(f"norm df: \n{norm_df}")
            gen_plot(norm_df,p,False,norm_pkl_name,func_name)


if __name__ == '__main__':
    main()
