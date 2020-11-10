import multiprocessing
import time
import numpy as np

def gen_time_results(mat_size,no_runs):
    mat_A = np.random.rand(mat_size,mat_size)
    mat_B = np.random.rand(mat_size,mat_size)
    time_list = []
    for _ in range(no_runs):
        start = time.perf_counter()
        result = np.matmul(mat_A,mat_B)
        finish = time.perf_counter()
        time_taken = round(finish-start,4)
        time_list.append(time_taken)
    return time_list


def main():
    size_list = [32,64,128,256,512,1024]
    for mat_size in size_list:
        print('Matrix size:')
        print(mat_size)
        no_runs = 1000
        time_results = gen_time_results(mat_size,no_runs)
        print('Run time:')
        print(sum(time_results)/len(time_results))


main()
