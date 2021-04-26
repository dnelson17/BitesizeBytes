from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

def read_python_times():
    time_df = pd.read_pickle("time_df_libraries_lock.pkl")
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    time_df.columns = ["Numpy matmul","Python dgemm","Python sgemm"]
    return time_df


def read_fortran_times(p):
    fortran_df = pd.read_fwf(f"{p.parent}\Fortran/fortran_results.txt")
    fortran_df.columns = ["NumProcs","MatSize","Times"]
    fortran_df = fortran_df.pivot(index="MatSize",columns="NumProcs",values="Times")
    fortran_df = fortran_df.drop([2,4,8,16,32],axis=1)
    fortran_df.columns = ["Fortran sgemm"]
    return fortran_df


def gen_cubed_df(start_power,end_power):
    order_list = [2**j for j in range(start_power,end_power)]
    cubed_list = [(2**j)**3 for j in range(start_power,end_power)]
    cubed_df = pd.DataFrame([cubed_list],columns=order_list,index=["(n^3)/100"])
    return cubed_df.T


def apply_norm(time_df,norm_index):
    norm_df = (time_df.T).apply(lambda x: x/x.iloc[norm_index], axis=1, result_type='expand')
    return norm_df


def gen_plot(df,p):
    df = df.T
    df.plot()
    plt.xlabel("Matrix Order")
    plt.ylabel("Normalised Runtime")
    plt.legend()
    #plt.gca().set_aspect('equal')
    p = Path.cwd()
    #plt.show()
    plt.savefig(f"{p.parent.parent}\Figures\Library_Normalised_Runtimes.png")


def main():
    p = Path.cwd()
    
    start_power = 5
    end_power = 16
    norm_index = 0
    
    time_df = read_python_times()
    print(f"time_df: \n{time_df}")
    fortran_df = read_fortran_times(p)
    print(f"fortran_df: \n{fortran_df}")
    total_df = time_df.join(fortran_df)
    print(f"total_df: \n{total_df}")
    cubed_df = gen_cubed_df(start_power,end_power)
    print(f"cubed_df: \n{cubed_df}")
    total_df = total_df.join(cubed_df)
    print(f"total_df: \n{total_df}")
    norm_df = apply_norm(total_df,norm_index)
    #norm_df = norm_df.drop(['My_function','My_function_(32_Cores)'])
    gen_plot(norm_df,p)


if __name__ == '__main__':
    main()
