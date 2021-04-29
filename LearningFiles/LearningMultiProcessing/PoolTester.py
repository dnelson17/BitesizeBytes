from multiprocessing import Pool
import time
import numpy as np

def check_answer(mat_A,mat_B,mat_C):
    answer = np.matmul(mat_A,mat_B)
    ans = np.array_equal(answer,mat_C)
    return ans


def matrix_mult(start, mat_A, mat_B):
    mat_C = np.matmul(mat_A, mat_B)
    res = np.zeros(mat_B.shape, dtype = int)
    res[start:mat_C.shape[0]+start,:mat_C.shape[1]] = mat_C
    return res


def gen_time_results(mat_size,max_cores,no_runs):
    tally = 0
    mat_A = np.random.randint(0,3,size=(mat_size,mat_size))
    mat_B = np.random.randint(3,6,size=(mat_size,mat_size))
    mat_C = np.zeros((mat_size,mat_size), dtype = int)
    time_mat = []
    for no_cores in range(1,max_cores+1):
        print(f'\nRunning on {no_cores} cores(s)\n')
        time_mat.append([])
        for _ in range(no_runs):
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
                time_taken = round(finish-start,4)
                time_mat[no_cores-1].append(time_taken)
                tally += check_answer(mat_A,mat_B,result)
    return time_mat, tally


def main():
    mat_size = 16
    print(f'Matrix size: {mat_size}')
    max_cores = 8
    no_runs = 10
    time_results, no_correct = gen_time_results(mat_size,max_cores,no_runs)
    no_incorrect = no_runs*max_cores - no_correct
    print(f'No correct: {no_correct},  No incorrect: {no_incorrect}')


main()

