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


def gen_plot(df,p,speedup):
    df = df.T
    df.plot()
    plt.xlabel("Number of processors (p)")
    if speedup:
        plt.ylabel("Runtime Speedup")
    else:
        plt.ylabel("Normalised Runtime")
    plt.legend([f"10^{i}" if i != "Ideal" else "Ideal" for i in df.columns])
    p = Path.cwd()
    if speedup:
        plt.savefig(f"{p.parent.parent}\Figures\MonteCarlo\MonteCarlo_strong_speedup.png")
    else:
        plt.savefig(f"{p.parent.parent}\Figures\MonteCarlo\MonteCarlo_weak_norm.png")


def main():
    max_cores = 6
    core_list = [2**i for i in range(max_cores)]
    p = Path.cwd()
    norm_index = 0
    
    strong_time_df = read_df_pickle(f"Time_dfs/time_df_strong_scaling.pkl")
    print(f"strong time df: \n{strong_time_df}")
    strong_speedup_df = apply_speedup(strong_time_df,core_list)
    print(f"speedup df: \n{strong_speedup_df}")
    gen_plot(strong_speedup_df,p,True)
    
    weak_time_df = read_df_pickle(f"Time_dfs/time_df_weak_scaling.pkl")
    print(f"weak time df: \n{weak_time_df}")
    weak_norm_df = apply_norm(weak_time_df,norm_index,core_list)
    print(f"norm df: \n{weak_norm_df}")
    gen_plot(weak_norm_df,p,False)


if __name__ == '__main__':
    main()
