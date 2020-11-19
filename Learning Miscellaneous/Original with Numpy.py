from multiprocessing import Process
import time
import numpy as np

def check_answer(mat_A,mat_B,mat_C):
    answer = np.matmul(mat_A,mat_B)
    comparison = mat_C == answer
    if comparison.all():
        return True
    else:
        return False


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
                    #first = round(i * mat_size / no_cores)
                    #last = round((i + 1) * mat_size / no_cores)
                    #param = [mat_A[first:last], mat_B[first:last]]
                    param = [round(i * mat_size / no_cores), round((i + 1) * mat_size / no_cores), mat_A, mat_B]
                    p = Process(target=matrix_mult, args=param)
                    p.start()
                    processes.append(p)
                for process in processes:
                    process.join()
                finish = time.perf_counter()
                time_taken = round(finish-start,4)
                time_mat[no_cores-1].append(time_taken)
                #tally += check_answer(mat_A,mat_B,result)
    return time_mat#, tally


def main():
    print('Matrix size:')
    print(mat_size)
    max_cores = 1
    no_runs = 1
    time_results = gen_time_results(mat_size,max_cores,no_runs)
    #gen_results_graph(time_results)


main()
