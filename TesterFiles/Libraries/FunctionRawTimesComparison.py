from multiprocessing import Pool, shared_memory
from scipy.linalg import blas as FB
import pandas as pd
import numpy as np
import time

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[1]))
    for i in range(len(mat_A)):
        for j in range(len(mat_B[i])):
            for k in range(len(mat_B)):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    return mat_C


def matrix_mult_parallel(i1,i_size,j1,j_size,mat_size,name_A,name_B):
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)[i1*i_size:(i1+1)*i_size,:]
    mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)[:,j1*j_size:(j1+1)*j_size]
    mat_C = np.zeros((i_size,j_size))
    for i in range(i_size):
        for j in range(j_size):
            for k in range(mat_size):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    existing_shm_A.close()
    existing_shm_B.close()
    return mat_C


def gen_time_results(mat_size, no_cores, data_A, data_B):
    mat_shape = (mat_size,mat_size)
    shm_A = shared_memory.SharedMemory(create=True, size=data_A.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=data_A.nbytes)
    mat_A = np.ndarray(data_A.shape, dtype=data_A.dtype, buffer=shm_A.buf)
    mat_B = np.ndarray(data_B.shape, dtype=data_B.dtype, buffer=shm_B.buf)
    name_A = shm_A.name
    name_B = shm_B.name
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
            send_list.append([i,i_size,j,j_size,mat_size,name_A,name_B])
    start = time.perf_counter()
    p = Pool(processes=no_cores)
    res_list = p.starmap(matrix_mult_parallel, send_list)
    p.close()
    result = np.vstack( np.split( np.concatenate(res_list,axis=1) , pars_i, axis=1) )
    finish = time.perf_counter()
    time_taken = round(finish-start,10)
    shm_A.close()
    shm_B.close()
    shm_A.unlink()
    shm_B.unlink()
    return time_taken, result


def main():
    size_list = [2**i for i in range(5,14)]
    no_runs = 10
    time_df = pd.DataFrame(columns=["My_function","My_function_(32_Cores)","Numpy_MatMul","Lapack_dgemm","Lapack_sgemm"])
    for mat_size in size_list:
        print(f"Mat size: {mat_size}")
        for i in range(no_runs):
            print(f"i: {i}")

            total_time_Numpy = 0
            m1 = np.random.rand(mat_size,mat_size).astype(np.float32)
            m2 = np.random.rand(mat_size,mat_size).astype(np.float32)
            new_times=[]

            time.sleep(10)
            
            if mat_size < 1024:
                my_func_start = time.perf_counter()
                m_myfunc = matrix_mult(m1,m2)
                my_func_finish = time.perf_counter()
                new_times.append(round(my_func_finish-my_func_start,8))

                time.sleep(5)
                
                my_func_32cores_start = time.perf_counter()
                time_taken, m_myfunc32 = gen_time_results(mat_size,32,m1,m2)
                my_func_32cores_finish = time.perf_counter()
                new_times.append(round(my_func_32cores_finish-my_func_32cores_start,8))

                time.sleep(5)
            else:
                new_times.append(None)
                new_times.append(None)
            
            numpy_start = time.perf_counter()
            mn = np.matmul(m1,m2)
            numpy_finish = time.perf_counter()
            new_times.append(round(numpy_finish-numpy_start,8))

            time.sleep(5)
            
            dgemm_start = time.perf_counter()
            md = FB.dgemm(alpha=1, a=m1, b=m2)
            dgemm_finish = time.perf_counter()
            new_times.append(round(dgemm_finish-dgemm_start,8))

            time.sleep(5)
            
            sgemm_start = time.perf_counter()
            ms = FB.sgemm(alpha=1, a=m1, b=m2)
            sgemm_finish = time.perf_counter()
            new_times.append(round(sgemm_finish-sgemm_start,8))

            time.sleep(5)

            print(new_times)
            
            time_df = time_df.append( pd.DataFrame([new_times],columns=["My_function","My_function_(32_Cores)","Numpy_MatMul","Lapack_dgemm","Lapack_sgemm"],index=[mat_size]) )
            time_df.to_pickle("time_df_libraries.pkl")
    print(f"\nOriginal times:\n{time_df}")
    time_df = time_df.sort_index()
    time_df = time_df.groupby(time_df.index).mean()
    print(f"\nTimes after ordering and mean:\n{time_df}")


if __name__ == '__main__':
    main()
