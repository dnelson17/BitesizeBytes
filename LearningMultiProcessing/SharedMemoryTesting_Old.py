from multiprocessing import Pool
from scipy.linalg import blas as FB
import pandas as pd
import numpy as np
import time

#python3 TestingWithLapack.py

def matrix_mult(mat_A, mat_B):
    calc_start = time.perf_counter()
    mat_C = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)
    calc_finish = time.perf_counter()
    return mat_C, calc_start, calc_finish


def gen_time_results(mat_size,core_list):
    mat_A = np.random.rand(mat_size,mat_size).astype(np.float32)
    mat_B = np.random.rand(mat_size,mat_size).astype(np.float32)
    total_times = []
    send_times = []
    calc_times = []
    recv_times = []
    for no_cores in core_list:
        print(no_cores)
        #Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
        power = np.log2(no_cores)/2
        #Represents the number of partitons that must be calculated in the result matrix C
        pars_i = int(2**(np.ceil(power)))
        pars_j = int(2**(np.floor(power)))
        #Represents the size of each partiton in the i and j axis
        i_size = int(mat_size/pars_i)
        j_size = int(mat_size/pars_j)
        send_list = []
        start = time.perf_counter()
        for i in range(pars_i):
            for j in range(pars_j):
                send_list.append([mat_A[i*i_size:(i+1)*i_size,:],mat_B[:,j*j_size:(j+1)*j_size]])
        p = Pool(processes=no_cores)
        res_list = p.starmap(matrix_mult, send_list)
        p.close()
        mat_res_list = [res_list[i][0] for i in range(len(res_list))]
        result = np.vstack( np.split( np.concatenate(mat_res_list,axis=1) , pars_i, axis=1) )
        finish = time.perf_counter()
        calc_start = 0
        calc_finish = 0
        for i in range(len(res_list)):
            calc_start += res_list[i][1]
            calc_finish += res_list[i][2]
        calc_start /= len(res_list)
        calc_finish /= len(res_list)
        send_times.append( round(calc_start-start,10) )
        calc_times.append( round(calc_finish-calc_start,10) )
        recv_times.append( round(finish-calc_finish,10) )
        total_times.append( round(finish-start,10) )
    return tuple(send_times), tuple(calc_times), tuple(recv_times), tuple(total_times)


def main():
    size_list = [2**i for i in range(5,18)]
    core_list = [2**j for j in range(6)]
    no_runs = 4
    send_time_df = pd.DataFrame(columns=core_list)
    calc_time_df = pd.DataFrame(columns=core_list)
    recv_time_df = pd.DataFrame(columns=core_list)
    total_time_df = pd.DataFrame(columns=core_list)
    for mat_size in size_list:
        for _ in range(no_runs):
            print(f"{mat_size}")
            send_times, calc_times, recv_times, total_times = gen_time_results(mat_size, core_list)
            print(f"send_times: {send_times},\ncalc_times: {calc_times},\nrecv_times: {recv_times},\ntotal_times: {total_times}")
            send_time_df = send_time_df.append( pd.DataFrame([send_times],columns=core_list,index=[mat_size]) )
            send_time_df.to_pickle("send_time_df_lapack_old.pkl")
            calc_time_df = calc_time_df.append( pd.DataFrame([calc_times],columns=core_list,index=[mat_size]) )
            calc_time_df.to_pickle("calc_time_df_lapack_old.pkl")
            recv_time_df = recv_time_df.append( pd.DataFrame([recv_times],columns=core_list,index=[mat_size]) )
            recv_time_df.to_pickle("recv_time_df_lapack_old.pkl")
            total_time_df = total_time_df.append( pd.DataFrame([total_times],columns=core_list,index=[mat_size]) )
            total_time_df.to_pickle("total_time_df_lapack_old.pkl")
        print("")
    print(f"\nOriginal times:\n{time_df}")
    send_time_df = send_time_df.sort_index()
    send_time_df = send_time_df.groupby(send_time_df.index).mean()
    send_time_df.to_pickle("send_time_df_lapack_old.pkl")
    calc_time_df = calc_time_df.sort_index()
    calc_time_df = calc_time_df.groupby(calc_time_df.index).mean()
    calc_time_df.to_pickle("calc_time_df_lapack_old.pkl")
    recv_time_df = recv_time_df.sort_index()
    recv_time_df = recv_time_df.groupby(recv_time_df.index).mean()
    recv_time_df.to_pickle("recv_time_df_lapack_old.pkl")
    total_time_df = total_time_df.sort_index()
    total_time_df = total_time_df.groupby(total_time_df.index).mean()
    total_time_df.to_pickle("total_time_df_lapack_old.pkl")


if __name__ == '__main__':
    main()
