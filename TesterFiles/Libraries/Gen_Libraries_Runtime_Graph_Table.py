from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import re

def read_python_times(path_name):
    time_df = pd.read_pickle(path_name)
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    time_df.columns = ["Numpy matmul","Python dgemm","Python sgemm"]
    return time_df


def read_fortran_times(path_name):
    fortran_df = pd.read_fwf(path_name)
    fortran_df.columns = ["NumProcs","MatSize","Times"]
    fortran_df = fortran_df.pivot(index="MatSize",columns="NumProcs",values="Times")
    fortran_df = fortran_df.drop([2,4,8,16,32],axis=1)
    fortran_df.columns = ["Fortran sgemm"]
    return fortran_df


def read_myfunc_times(path_name):
    time_df = pd.read_pickle(path_name)
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    return time_df


def gen_cubed_df(start_power,end_power):
    order_list = [2**j for j in range(start_power,end_power)]
    cubed_list = [(2**j)**3 for j in range(start_power,end_power)]
    cubed_df = pd.DataFrame([cubed_list],columns=order_list,index=["n^3"])
    return cubed_df.T


def apply_norm(time_df,norm_index):
    norm_df = (time_df.T).apply(lambda x: x/x.iloc[norm_index], axis=1, result_type='expand')
    return norm_df


def gen_plot(df,p,lock_val):
    df = df.T
    df.plot()
    plt.xlabel("Matrix Order (N)")
    plt.ylabel("Normalised Runtime (T)")
    plt.legend()
    #plt.gca().set_aspect('equal')
    p = Path.cwd()
    plt.show()
    #plt.savefig(f"{p.parent.parent}\Figures\Libraries\Library_Normalised_Runtimes_{lock_val}.png")


def calc_FLOPS(df):
    flops_df = df.apply(lambda x: (x.name**3)/(x*(10**9)), axis=1, result_type='expand')
    return flops_df


def print_overleaf_table(df):
    df = df.round(decimals=4)
    df.columns = ["My_function_(Python)","My_function_(32_Cores_Python)",
                  "matmul_(NumPy_Python)","dgemm_(Python)","sgemm_(Python)","sgemm_(Fortran)"]
    df = df.sort_index()
    df = df.groupby(df.index).mean()
    df_string = df.to_string().replace("\n","\\\_\n")
    df_string = re.sub(' +', ' ', df_string)
    df_string = df_string.replace(" ","&")
    df_string = df_string.replace("_"," ")
    df_string+="\\\\"
    print(df_string)


def main():
    p = Path.cwd()
    
    start_power = 8
    end_power = 16
    norm_index = 0

    for lock_val in ["no_lock","lock"]:
        print(f"\n\n{lock_val}\n")
        python_df = read_python_times(f"time_df_libraries_{lock_val}.pkl")
        print(f"python_df: \n{python_df}")
        
        fortran_df = read_fortran_times(f"{p.parent}\Fortran/fortran_results_{lock_val}.txt")
        print(f"fortran_df: \n{fortran_df}")
        python_and_fortran_df = python_df.join(fortran_df)
        
        cubed_df = gen_cubed_df(start_power,end_power)
        print(f"cubed_df: \n{cubed_df}")
        
        total_df = python_and_fortran_df.join(cubed_df)
        print(f"total_df: \n{total_df}")
        norm_df = apply_norm(total_df,norm_index)
        print(f"norm_df: \n{norm_df.to_string()}")
    
        gen_plot(norm_df,p,lock_val)
    myfunc_df = read_myfunc_times(f"time_df_my_function_{lock_val}.pkl")
    python_and_fortran_df = myfunc_df.join(python_and_fortran_df)
    print_overleaf_table(python_and_fortran_df)
    print(calc_FLOPS(python_and_fortran_df).to_string())


if __name__ == '__main__':
    main()
