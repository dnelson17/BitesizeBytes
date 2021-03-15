from multiprocessing import Pool, shared_memory
from scipy.linalg import blas as FB
import pandas as pd
import numpy as np
import time

def matrix_mult(i1,i2,j1,j2,mat_size,name_A,name_B):
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)
    mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)
    mat_C = FB.sgemm(alpha=1.0, a=mat_A[i1:i2,:], b=mat_B[:,j1:j2])
    existing_shm_A.close()
    existing_shm_B.close()
    return mat_C


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
    times = []
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
        for i in range(pars_i):
            for j in range(pars_j):
                send_list.append([i*i_size,(i+1)*i_size,j*j_size,(j+1)*j_size,mat_size,name_A,name_B])
        start = time.perf_counter()
        p = Pool(processes=no_cores)
        res_list = p.starmap(matrix_mult, send_list)
        p.close()
        finish = time.perf_counter()
        result = np.vstack( np.split( np.concatenate(res_list,axis=1) , pars_i, axis=1) )
        time_taken = round(finish-start,10)
        print(time_taken)
        times.append(time_taken)
    shm_A.close()
    shm_B.close()
    shm_A.unlink()
    shm_B.unlink()
    return tuple(times)


def main():
    size_list = [2**i for i in range(5,18)]
    core_list = [2**j for j in range(6)]
    no_runs = 4
    time_df = pd.DataFrame(columns=core_list)
    for mat_size in size_list:
        for _ in range(no_runs):
            print(f"{mat_size}")
            new_times = gen_time_results(mat_size, core_list)
            print(new_times)
            time_df = time_df.append( pd.DataFrame([new_times],columns=core_list,index=[mat_size]) )
            time_df.to_pickle("time_df_lapack.pkl")
        print("")
    print(f"\nOriginal times:\n{time_df}")
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    print(f"\nTimes after ordering and mean:\n{time_df}")
    speedup_df = time_df.apply(lambda x: x.iloc[0]/x, axis=1, result_type='expand')
    ideal_df = pd.DataFrame([core_list],columns=core_list,index=["Ideal"])
    print(f"\nIdeal:\n{ideal_df}")
    speedup_df = speedup_df.append( ideal_df )
    print(f"\nSpeedups:\n{speedup_df}")
    time_df.to_pickle("time_df_lapack.pkl")
    speedup_df.to_pickle("speedup_df_lapack.pkl")
    

if __name__ == '__main__':
    main()
