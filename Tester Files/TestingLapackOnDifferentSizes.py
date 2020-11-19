import time
import numpy as np
from scipy.linalg import blas as FB

def gen_time_results(mat_size,no_runs):
    mat_A = np.random.rand(mat_size,mat_size)
    mat_B = np.random.rand(mat_size,mat_size)
    time_list_d = []
    time_list_s = []
    for _ in range(no_runs):
        start = time.perf_counter()
        result = FB.dgemm(alpha=1, a=mat_A, b=mat_B)
        finish = time.perf_counter()
        time_taken_d = round(finish-start,10)
        time_list_d.append(time_taken_d)
        
        start = time.perf_counter()
        result = FB.sgemm(alpha=1, a=mat_A, b=mat_B)
        finish = time.perf_counter()
        time_taken_s = round(finish-start,10)
        time_list_s.append(time_taken_s)
    return time_list_d, time_list_s


def main():
    size_list = [32,64,128,256,512,1024]
    for mat_size in size_list:
        print('Matrix size:')
        print(mat_size)
        no_runs = 1000
        time_results_d, time_results_s = gen_time_results(mat_size,no_runs)
        print('Run time (d):')
        print(sum(time_results_d)/len(time_results_d))
        print('Run time (s):')
        print(sum(time_results_s)/len(time_results_s))


main()
