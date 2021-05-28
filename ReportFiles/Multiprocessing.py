from multiprocessing import Pool, shared_memory
from scipy.linalg import blas as FB
import pandas as pd
import numpy as np
import time

#Sgemm version
def matrix_mult(i,len_i,j,len_j,mat_size,name_A,name_B,name_C):
    stabiliser_time = time.time()
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    existing_shm_C = shared_memory.SharedMemory(name=name_C)
    i1 = i*len_i
    i2 = (i+1)*len_i
    j1 = j*len_j
    j2 = (j+1)*len_j
    sub_mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)[i1:i2,:]
    sub_mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)[:,j1:j2]
    sub_mat_C = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_C.buf)
    calc_start = time.time()
    sub_mat_C[i1:i2,j1:j2] = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B)
    calc_finish = time.time()
    existing_shm_A.close()
    existing_shm_B.close()
    existing_shm_C.close()
    return stabiliser_time, calc_start, calc_finish


#MyFunc version
def matrix_mult(i,len_i,j,len_j,mat_size,name_A,name_B,name_C):
    stabiliser_time = time.time()
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    existing_shm_C = shared_memory.SharedMemory(name=name_C)
    i1 = i*len_i
    i2 = (i+1)*len_i
    j1 = j*len_j
    j2 = (j+1)*len_j
    sub_mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)
    sub_mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)
    sub_mat_C = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_C.buf)
    calc_start = time.time()
    for i in range(i1,i2):
        for j in range(j1,j2):
            for k in range(mat_size):
                sub_mat_C[i,j] += sub_mat_A[i,k] * sub_mat_B[k,j]
    calc_finish = time.time()
    existing_shm_A.close()
    existing_shm_B.close()
    existing_shm_C.close()
    return stabiliser_time, calc_start, calc_finish


def gen_time_results(mat_size, core_list):
    mat_shape = (mat_size,mat_size)
    data_A = np.random.rand(*mat_shape).astype(np.float32)
    data_B = np.random.rand(*mat_shape).astype(np.float32)
    data_C = np.empty((mat_size,mat_size),dtype=np.float32)
    shm_A = shared_memory.SharedMemory(create=True, size=data_A.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=data_B.nbytes)
    shm_C = shared_memory.SharedMemory(create=True, size=data_C.nbytes)
    mat_A = np.ndarray(data_A.shape, dtype=data_A.dtype, buffer=shm_A.buf)
    mat_A[:] = data_A[:]
    mat_B = np.ndarray(data_B.shape, dtype=data_B.dtype, buffer=shm_B.buf)
    mat_B[:] = data_B[:]
    mat_C = np.ndarray(data_C.shape, dtype=data_C.dtype, buffer=shm_C.buf)
    mat_C[:] = data_C[:]
    name_A = shm_A.name
    name_B = shm_B.name
    name_C = shm_C.name
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
        len_i = int(mat_size/pars_i)
        len_j = int(mat_size/pars_j)
        send_list = []
        total_start = time.time()
        send_list = [[i,len_i,j,len_j,mat_size,name_A,name_B,name_C] for j in range(pars_j) for i in range(pars_i)]
        p = Pool(processes=no_cores)
        res_list = p.starmap(matrix_mult, send_list)
        p.close()
        total_finish = time.time()
        calc_start_list = []
        calc_finish_list = []
        res_list = list(res_list)
        for i in range(len(res_list)):
            time_difference = res_list[0][0] - res_list[i][0]
            calc_start_list.append(res_list[i][1]+time_difference)
            calc_finish_list.append(res_list[i][2]+time_difference)
        calc_start = min(calc_start_list)
        calc_finish = max(calc_finish_list)
        send_time = calc_start-total_start
        calc_time = calc_finish-calc_start
        gather_time = total_finish-calc_finish
        total_time = total_finish-total_start
        assert send_time + calc_time + gather_time == total_time
        send_times.append( round(send_time,10) )
        calc_times.append( round(calc_time,10) )
        recv_times.append( round(gather_time,10) )
        total_times.append( round(total_time,10) )
    shm_A.close()
    shm_B.close()
    shm_C.close()
    shm_A.unlink()
    shm_B.unlink()
    shm_C.unlink()
    return tuple(send_times), tuple(calc_times), tuple(recv_times), tuple(total_times)


def main():
    size_list = [2**i for i in range(7,17)]
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
            send_time_df.to_pickle("Time_dfs/scatter_df.pkl")
            calc_time_df = calc_time_df.append( pd.DataFrame([calc_times],columns=core_list,index=[mat_size]) )
            calc_time_df.to_pickle("Time_dfs/calc_df.pkl")
            recv_time_df = recv_time_df.append( pd.DataFrame([recv_times],columns=core_list,index=[mat_size]) )
            recv_time_df.to_pickle("Time_dfs/gather_df.pkl")
            total_time_df = total_time_df.append( pd.DataFrame([total_times],columns=core_list,index=[mat_size]) )
            total_time_df.to_pickle("Time_dfs/total_df.pkl")
        print("")


if __name__ == '__main__':
    main()
