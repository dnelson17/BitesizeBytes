#This includes code that I will use in my overleaf doc to show how a matrix
#multiplication program will scale across a number of cores from 1-8
#and as the size of the matrix grows
import multiprocessing
import time
import numpy as np
#from matplotlib import pyplot as plt

mat_size = 840 #chosen to be 8! so the work can be divided up nicely for any number of cores from 1-8
mat1 = np.random.randint(10, size=(mat_size,mat_size))
mat2 = np.random.randint(10, size=(mat_size,mat_size))
mat3 = np.zeros((mat_size,mat_size), dtype = int)
time_list = [] #list for the total time taken for each no of cores

#the start&end parameters will tell the function what row to begin and finish on
def matrix_mult(first_row, last_row, mat_A, mat_B, mat_C):
    for i in range(first_row, last_row):
        for j in range(mat_size):
            for k in range(mat_size):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    print(mat_C)
    return mat_C


for no_cores in range(1,9):
    result = mat3
    
    start = time.perf_counter()
    
    processes = []
    
    if __name__ == '__main__':
        for i in range(no_cores):
            param = [int(i * mat_size / no_cores), int((i + 1) * mat_size / no_cores), mat1, mat2, result]
            p = multiprocessing.Process(target=matrix_mult, args=param)
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        finish = time.perf_counter()

        time_taken = round(finish-start,4)
        print(f'{no_cores} cores(s) - Finished in {time_taken} seconds')

        time_list.append(time_taken)

        #answer = np.matmul(mat1,mat2)

        #comparison = result == answer
        #equal_arrays = comparison.all()

        #if equal_arrays:
        #    print('correct')

print(time_list)


#speedup = []

#for i in range(len(results)):
#  speedup.append( origrinal_time/results[i] )

#print(speedup)

#plt.plot([1,2,3,4,5,6,7,8],speedup)
#plt.xlabel("No of cores")
#plt.ylabel("Speed-up")
#plt.show()
#plt.plot([1,2,3,4,5,6,7,8],time_list)
#plt.show()
