from multiprocessing import Pool
import time
import numpy as np
from scipy.linalg import blas as FB

#python3 main_tester_lapack.py

def check_answer(mat_A,mat_B,mat_C):
    answer = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)
    #rounded_answer = np.around(answer,decimals=5)
    #rounded_mat_C = np.around(mat_C,decimals=5)
    return np.allclose(answer,mat_C)


def matrix_mult(start, mat_A, mat_B):
    mat_C = FB.sgemm(alpha=1.0, a=mat_A, b=mat_B)
    res = np.zeros(mat_B.shape)
    res[start:mat_C.shape[0]+start,:mat_C.shape[1]] = mat_C
    return res


def gen_time_results(mat_size,max_cores,no_runs):
    tally = 0
    time_mat = []
    for no_cores in range(1,max_cores+1):
        print(f"{no_cores} cores(s)")
        time_mat.append([])
        for _ in range(no_runs):
            mat_A = np.random.rand(mat_size,mat_size)
            mat_B = np.random.rand(mat_size,mat_size)
            result = mat_C
            start = time.perf_counter()
            param = []
            if __name__ == '__main__':
                for i in range(no_cores):
                    first = round(i * mat_size / no_cores)
                    last = round((i + 1) * mat_size / no_cores)
                    param.append((first,mat_A[first:last,:],mat_B))
                p = Pool(processes=no_cores)
                data = p.starmap(matrix_mult, param)
                p.close()
                result = sum(data)
                finish = time.perf_counter()
                time_taken = round(finish-start,10)
                time_mat[no_cores-1].append(time_taken)
                tally += check_answer(mat_A,mat_B,result)
    return time_mat, tally

def gen_results_graph(time_mat):
    time_mat = np.array(time_mat)
    av_time_array = time_mat.mean(axis = 1)
    av_time_list = av_time_array.tolist()
    print('Average times:')
    print(av_time_list)
    origrinal_time = av_time_list[0]
    my_speedup = []
    no_cores = []
    for i in range(1,len(av_time_list)+1):
        my_speedup.append( origrinal_time/av_time_list[i-1] )
        no_cores.append( i )
    print('Average speedups:')
    print(my_speedup)


def main():
    size_list = [32,64,128,256,512,1024]#,2048,4096,8192]
    total = 0
    for mat_size in size_list:
        print(f"Matrix size: {mat_size}")
        max_cores = 32
        no_runs = 10
        time_results, no_correct = gen_time_results(mat_size,max_cores,no_runs)
        gen_results_graph(time_results)
        no_incorrect = no_runs*max_cores - no_correct
        print(f'No correct: {no_correct},  No incorrect: {no_incorrect}')
        total += no_correct
    total_incorrect = no_runs*max_cores*len(size_list) - total
    print(f'Total correct: {total},  Total incorrect: {total_incorrect}')


main()

