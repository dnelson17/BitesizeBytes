from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import re

def read_python_times(path_name):
    time_df = pd.read_pickle(path_name)
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    #time_df.columns = ["Numpy matmul","Python dgemm","Python sgemm"]
    return time_df


def read_fortran_times(path_name):
    fortran_df = pd.read_fwf(path_name)
    fortran_df.columns = ["NumProcs","MatSize","Times"]
    fortran_df = fortran_df.pivot(index="MatSize",columns="NumProcs",values="Times")
    fortran_df = fortran_df.drop([1,2,4,8,16],axis=1)
    fortran_df.columns = ["Fortran sgemm"]
    return fortran_df


def gen_plot(df,p):
    #df = df.T
    print(df.columns)
    df.plot()
    plt.xlabel("Matrix Order (N)")
    plt.ylabel("Runtime (T)")
    plt.legend()
    #plt.gca().set_aspect('equal')
    p = Path.cwd()
    #plt.show()
    #plt.savefig(f"{p.parent.parent}\Figures\Libraries\Comparing_Parallel_Runtimes.png")


def main():
    p = Path.cwd()
    
    start_power = 8
    end_power = 16
    norm_index = 0

    print("\n\n")
    no_lock_df = read_python_times(f"time_df_libraries_no_lock.pkl")
    no_lock_df = no_lock_df.drop(["Python dgemm (No Lock)","Numpy MatMul (No Lock)"],axis=1)
    no_lock_df.columns = ["Python Multithreading"]
    print(f"no_lock_df: \n{no_lock_df}")

    multiprocessing_df = read_python_times(f"{p.parent}\Multiprocessing/Lapack/Time_dfs/total_df.pkl")
    multiprocessing_df = multiprocessing_df.drop([1,2,4,8,16],axis=1)
    print(multiprocessing_df.columns)
    multiprocessing_df.columns = ["Python Multiprocessing (32 Cores)"]
    print(f"multiprocessing_df: \n{multiprocessing_df}")
    total_df = no_lock_df.join(multiprocessing_df)

    mpi_df = read_python_times(f"{p.parent}\MPI/Lapack/Time_dfs/total_df.pkl")
    mpi_df = mpi_df.drop([1,2,4,8,16],axis=1)
    mpi_df.columns = ["Python MPI (32 Cores)"]
    print(f"mpi_df: \n{mpi_df}")
    total_df = total_df.join(mpi_df)
    
    fortran_df = read_fortran_times(f"{p.parent}\Fortran/fortran_results.txt")
    fortran_df.columns = ["Fortran MPI (32 Cores)"]
    print(f"fortran_df: \n{fortran_df}")
    total_df = total_df.join(fortran_df)

    print(f"total_df: \n{total_df.to_string()}")

    gen_plot(total_df,p)


if __name__ == '__main__':
    main()
