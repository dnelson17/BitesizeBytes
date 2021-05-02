from matplotlib import pyplot as plt
import pandas as pd

def read_fortran_times():
    fortran_df = pd.read_fwf("fortran_results_lock.txt")
    fortran_df.columns = ["NumProcs","MatSize","Times"]
    fortran_df = fortran_df.pivot(index="MatSize",columns="NumProcs",values="Times")
    return fortran_df


def apply_speedup(time_df,core_list):
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    speedup_df = speedup_df.append( ideal_df )
    return speedup_df


def gen_plot(df):
    df = df.T
    df.plot()
    plt.xlabel("Number of Processors")
    plt.ylabel("Runtime Speedup")
    plt.legend()
    plt.show()


def main():
    max_cores = 6
    core_list = [2**i for i in range(max_cores)]
    fortran_df = read_fortran_times()
    print(f"fortran_df:\n{fortran_df}")
    speedup_df = apply_speedup(fortran_df,core_list)
    print(f"speedup_df:\n{speedup_df}")
    gen_plot(speedup_df)


if __name__ == '__main__':
    main()
