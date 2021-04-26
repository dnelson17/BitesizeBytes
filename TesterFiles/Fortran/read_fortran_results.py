from matplotlib import pyplot as plt
import pandas as pd

def read_fortran_times():
    fortran_df = pd.read_fwf("fortran_results.txt")
    fortran_df.columns = ["NumProcs","MatSize","Times"]
    fortran_df = fortran_df.pivot(index="MatSize",columns="NumProcs",values="Times")
    fortran_df = fortran_df.drop([2,4,8,16,32],axis=1)
    fortran_df.columns = ["Fortran"]
    return fortran_df


def gen_cubed_df(start_power,end_power):
    order_list = [2**j for j in range(start_power,end_power)]
    cubed_list = [(2**j)**3 for j in range(start_power,end_power)]
    cubed_df = pd.DataFrame([cubed_list],columns=order_list,index=["(n^3)/100"])
    return cubed_df.T


def apply_norm(time_df,norm_index):
    norm_df = (time_df.T).apply(lambda x: x/x.iloc[norm_index], axis=1, result_type='expand')
    return norm_df


def gen_plot(df):
    df = df.T
    df.plot()
    plt.xlabel("Matrix Order")
    plt.ylabel("Normalised Runtime")
    plt.legend()
    plt.show()


def main():
    start_power = 10
    end_power = 16
    norm_index = 0
    
    fortran_df = read_fortran_times()
    cubed_df = gen_cubed_df(start_power,end_power)
    total_df = fortran_df.join(cubed_df)
    norm_df = apply_norm(total_df,norm_index)
    print(norm_df)
    gen_plot(norm_df)


if __name__ == '__main__':
    main()
