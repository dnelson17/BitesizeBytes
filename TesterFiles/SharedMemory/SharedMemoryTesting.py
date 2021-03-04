from multiprocessing import Pool
import time
import numpy as np
from scipy.linalg import blas as FB

#python3 TestingWithLapack.py

def matrix_mult(mat_A, mat_B):
    mat_C = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)
    return mat_C


def gen_time_results(mat_size,max_cores,no_runs):
    for _ in range(no_runs):
        mat_A = np.random.rand(mat_size,mat_size)
        mat_B = np.random.rand(mat_size,mat_size)
        for no_cores in [1,2,4,8]:
            print(no_cores)
            #Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
            power = np.log2(no_cores)/2
            #Represents the number of partitons that must be calculated in the result matrix C
            i_len = int(2**(np.ceil(power)))
            j_len = int(2**(np.floor(power)))
            #Represents the size of each partiton in the i and j axis
            i_size = int(mat_size/i_len)
            j_size = int(mat_size/j_len)
            start = time.perf_counter()
            param = []
            if __name__ == '__main__':
                send_list = []
                for i in range(i_len):
                    for j in range(j_len):
                        send_list.append([mat_A[i*i_size:(i+1)*i_size,:],mat_B[:,j*j_size:(j+1)*j_size]])
                p = Pool(processes=no_cores)
                res_list = p.starmap(matrix_mult, send_list)
                p.close()
                result = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
                finish = time.perf_counter()
                time_taken = round(finish-start,10)
                print(time_taken)
    print("")
    return None


def main():
    size_list = [32,64,128,256]
    #size_list = [32,64,128,256,512,1024]
    #size_list = [2048,4096,8192,16384,32768]
    total = 0
    for mat_size in size_list:
        print(f"{mat_size}")
        max_cores = 8
        no_runs = 3
        gen_time_results(mat_size,max_cores,no_runs)


main()
