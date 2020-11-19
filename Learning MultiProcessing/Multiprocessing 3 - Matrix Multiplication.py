#This includes code that I will use in my overleaf doc to show how a matrix
#multiplication program will scale across a number of cores from 1-8
#and as the size of the matrix grows
import multiprocessing
import time
import numpy as np
#from matplotlib import pyplot as plt


#the start&end parameters will tell the function what row to begin and finish on
def matrix_mult(first_row, last_row, mat_size, mat_A, mat_B, mat_C):
    for i in range(first_row, last_row):
        for j in range(mat_size):
            for k in range(mat_size):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    return mat_C


def check_answer(mat_A,mat_B,mat_C):
    answer = np.matmul(mat_A,mat_B)
    comparison = mat_C == answer
    if comparison.all():
        return True
    else:
        return False


def gen_time_results(mat_size,max_cores,no_runs):
    tally = 0 #Tracker of how many calculations are correct
    
    mat_A = np.random.randint(10, size=(mat_size,mat_size))
    mat_B = np.random.randint(10, size=(mat_size,mat_size))
    mat_C = np.zeros((mat_size,mat_size), dtype = int)
    time_mat = []
    for no_cores in range(1,max_cores+1):
        print(f'Running on {no_cores} cores(s)')
        time_mat.append([])
        for _ in range(no_runs):
                result = mat_C
                
                start = time.perf_counter()
                
                processes = []
                
                if __name__ == '__main__':
                    for i in range(no_cores):
                        param = [round(i * mat_size / no_cores), round((i + 1) * mat_size / no_cores), mat_size, mat_A, mat_B, result]
                        p = multiprocessing.Process(target=matrix_mult, args=param)
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
    mat_size = 84000 #chosen to be 8! so the work can be divided up nicely for any number of cores from 1-8
    max_cores = 40
    no_runs = 20 #the code will run on each number of cores this many times
    
    #time_results = gen_time_results(mat_size,max_cores,no_runs)

    #no_incorrect = no_runs*max_cores - no_correct
    #print(f'No correct: {no_correct},  No incorrect: {no_incorrect}')

    #gen_time_file(time_results)


main()
    
