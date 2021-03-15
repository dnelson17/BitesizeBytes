from multiprocessing import Pool, RawArray
import time
import numpy as np
from scipy.linalg import blas as FB

# cd C:\University\Project\BitesizeBytes\TesterFiles\Multiprocessing
# python3 SharedMemoryTestingMyFunc.py

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(mat_A, mat_B):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['A'] = mat_A
    var_dict['B'] = mat_B


def matrix_mult(i1,i_size,j1,j_size,mat_size):
    mat_C = np.zeros((i_size,j_size))
    A_np = np.frombuffer(var_dict['A'],dtype=np.float32).reshape((mat_size,mat_size))[i1*i_size:(i1+1)*i_size,:]
    B_np = np.frombuffer(var_dict['B'],dtype=np.float32).reshape((mat_size,mat_size))[:,j1*j_size:(j1+1)*j_size]
    for i in range(i_size):
        for j in range(j_size):
            for k in range(mat_size):
                mat_C[i,j] += A_np[i][k] * B_np[k][j]
    return mat_C


def gen_time_results(mat_size, core_list, no_runs):
    if __name__ == '__main__':
        for _ in range(no_runs):
            mat_shape = (mat_size,mat_size)
            data_A = np.random.rand(*mat_shape).astype(np.float32)
            data_B = np.random.rand(*mat_shape).astype(np.float32)
            A = RawArray( 'f' , mat_shape[0] * mat_shape[1] )
            B = RawArray( 'f' , mat_shape[0] * mat_shape[1] )
            A_np = np.frombuffer( A, dtype = np.float32 ).reshape(mat_shape)
            B_np = np.frombuffer( B, dtype = np.float32 ).reshape(mat_shape)
            np.copyto(A_np, data_A)
            np.copyto(B_np, data_B)
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
                start = time.perf_counter()
                send_list = []
                for i in range(pars_i):
                    for j in range(pars_j):
                        send_list.append([i,i_size,j,j_size,mat_size])
                p = Pool(processes=no_cores, initializer=init_worker, initargs=(A, B))
                res_list = p.starmap(matrix_mult, send_list)
                p.close()
                result = np.vstack( np.split( np.concatenate(res_list,axis=1) , pars_i, axis=1) )
                finish = time.perf_counter()
                answer = np.matmul(data_A,data_B)
                #print(f"my result:\n{result}")
                #print(f"answer:\n{answer}")
                print(np.allclose(result,answer))
                time_taken = round(finish-start,10)
                print(time_taken)
                print("")
    print("")
    return None


def main():
    size_list = [2**i for i in range(5,13)]
    core_list = [2**i for i in range(4)]
    no_runs = 1
    for mat_size in size_list:
        print(f"{mat_size}")
        gen_time_results( mat_size, core_list, no_runs )


main()
