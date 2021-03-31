from multiprocessing import Pool, shared_memory
import pandas as pd
import numpy as np
import time

def matrix_mult(i1,i_size,j1,j_size,mat_size,name_A,name_B):
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)[i1*i_size:(i1+1)*i_size,:]
    mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)[:,j1*j_size:(j1+1)*j_size]
    mat_C = np.zeros((i_size,j_size))
    calc_start = time.perf_counter()
    for i in range(i_size):
        for j in range(j_size):
            for k in range(mat_size):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    calc_finish = time.perf_counter()
    existing_shm_A.close()
    existing_shm_B.close()
    return mat_C, calc_start, calc_finish


def gen_time_results(mat_size, core_list):
    mat_shape = (mat_size,mat_size)
    data_A = np.random.rand(*mat_shape).astype(np.float32)
    data_B = np.random.rand(*mat_shape).astype(np.float32)
    shm_A = shared_memory.SharedMemory(create=True, size=data_A.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=data_A.nbytes)
    mat_A = np.ndarray(data_A.shape, dtype=data_A.dtype, buffer=shm_A.buf)
    mat_B = np.ndarray(data_B.shape, dtype=data_B.dtype, buffer=shm_B.buf)
    name_A = shm_A.name
    name_B = shm_B.name
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
                send_list.append([i,i_size,j,j_size,mat_size,name_A,name_B])
        p = Pool(processes=no_cores)
        res_list = p.starmap(matrix_mult, send_list)
        p.close()
        mat_res_list = [res_list[i][0] for i in range(len(res_list))]
        result = np.vstack( np.split( np.concatenate(mat_res_list,axis=1) , pars_i, axis=1) )
        finish = time.perf_counter()
        calc_start_list = []
        calc_finish_list = []
        for i in range(len(res_list)):
            calc_start_list.append(res_list[i][1])
            calc_finish_list.append(res_list[i][2])
        calc_start = min(calc_start_list)
        calc_finish = max(calc_finish_list)
        send_times.append( round(calc_start-start,10) )
        calc_times.append( round(calc_finish-calc_start,10) )
        recv_times.append( round(finish-calc_finish,10) )
        total_times.append( round(finish-start,10) )
    shm_A.close()
    shm_B.close()
    shm_A.unlink()
    shm_B.unlink()
    return tuple(send_times), tuple(calc_times), tuple(recv_times), tuple(total_times)


def main():
    size_list = [2**i for i in range(5,12)]
    core_list = [2**j for j in range(6)]
    no_runs = 10
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
            send_time_df.to_pickle("send_time_df_myfunc.pkl")
            calc_time_df = calc_time_df.append( pd.DataFrame([calc_times],columns=core_list,index=[mat_size]) )
            calc_time_df.to_pickle("calc_time_df_myfunc.pkl")
            recv_time_df = recv_time_df.append( pd.DataFrame([recv_times],columns=core_list,index=[mat_size]) )
            recv_time_df.to_pickle("recv_time_df_myfunc.pkl")
            total_time_df = total_time_df.append( pd.DataFrame([total_times],columns=core_list,index=[mat_size]) )
            total_time_df.to_pickle("total_time_df_myfunc.pkl")
        print("")
    print(f"\nOriginal times:\n{time_df}")
    send_time_df = send_time_df.sort_index()
    send_time_df = send_time_df.groupby(send_time_df.index).mean()
    send_time_df.to_pickle("send_time_df_myfunc.pkl")
    calc_time_df = calc_time_df.sort_index()
    calc_time_df = calc_time_df.groupby(calc_time_df.index).mean()
    calc_time_df.to_pickle("calc_time_df_myfunc.pkl")
    recv_time_df = recv_time_df.sort_index()
    recv_time_df = recv_time_df.groupby(recv_time_df.index).mean()
    recv_time_df.to_pickle("recv_time_df_myfunc.pkl")
    total_time_df = total_time_df.sort_index()
    total_time_df = total_time_df.groupby(total_time_df.index).mean()
    total_time_df.to_pickle("total_time_df_myfunc.pkl")
    

if __name__ == '__main__':
    main()
