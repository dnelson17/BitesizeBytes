#This includes code that I will use in my overleaf doc to show how a matrix
#multiplication program will scale across a number of cores from 1-8
#and as the size of th3e matrix grows
import multiprocessing
import time
import numpy as np

mat_size = 40320 #chosen to be 8! so the work can be divided up nicely for any number of cores
mat1 = np.random.rand(mat_size,mat_size)
mat2 = np.random.rand(mat_size,mat_size)
mat3 = np.zeros((mat_size,mat_size), dtype = int)
time_list = [] #list for the total time taken for each no of cores

#the start&end parameters will tell the function what row to begin and finish on
def matrix_mult(start,end,result):
    for i in range(start,end):
        for j in range(mat_size):
            for k in range(mat_size):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result


for no_cores in range(1,9):
    result = mat3
    
    start = time.perf_counter()

    processes = []

    if __name__ == '__main__':
        for i in range(no_cores):
            param = [int(i* mat_size/no_cores), int((i+1)* mat_size/no_cores),result]
            p = multiprocessing.Process(target=matrix_mult, args=param )
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        finish = time.perf_counter()

        time_taken = round(finish-start,4)
        print(f'{no_cores} cores(s) - Finished in {time_taken} seconds')

        time_list.append(time_taken)

print(time_list)

# !!! Put in something to check that its actually correct, like do it with numpy and check results
