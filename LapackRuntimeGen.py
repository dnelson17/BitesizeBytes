import multiprocessing
import time
import numpy as np
from scipy.linalg import blas as FB

def check_answer(mat_A,mat_B,mat_C):
    answer = np.matmul(mat_A,mat_B)
    return mat_C.all() == answer()


def matrix_mult(first, last, mat_A, mat_B):
    print("Mat A: ")
    print(mat_A[first:last])
    mat_C = np.matmul(mat_A[first:last,:], mat_B[:,first:last])
    print(mat_C)
    return mat_C


def gen_time_results(mat_size,max_cores,no_runs):
    tally = 0
    mat_A = np.random.rand(mat_size,mat_size)
    mat_B = np.random.rand(mat_size,mat_size)
    mat_C = np.zeros((mat_size,mat_size), dtype = int)
    time_mat = []
    for no_cores in range(1,max_cores+1):
        #print(f'Running on {no_cores} cores(s)')
        print(no_cores)
        time_mat.append([])
        for _ in range(no_runs):
            result = mat_C
            start = time.perf_counter()
            processes = []
            if __name__ == '__main__':
                for i in range(no_cores):
                    first = round(i * mat_size / no_cores)
                    last = round((i + 1) * mat_size / no_cores)
                    param = [1,mat_A[first:last,:],mat_B]
                    p = multiprocessing.Process(target=FB.sgemm, args=param)
                    p.start()
                    processes.append(p)
                for process in processes:
                    process.join()
                finish = time.perf_counter()
                time_taken = round(finish-start,10)
                time_mat[no_cores-1].append(time_taken)
                #tally += check_answer(mat_A,mat_B,result)
    return time_mat#, tally

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
    size_list = [32,64,128,256,512,1024]
    for mat_size in size_list:
        print('Matrix size:')
        print(mat_size)
        max_cores = 32
        no_runs = 10
        time_results = gen_time_results(mat_size,max_cores,no_runs)
        gen_results_graph(time_results)


main()
