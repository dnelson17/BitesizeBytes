#This includes code that I will use in my overleaf doc to show how a matrix
#multiplication program will scale across a number of cores from 1-8
#and as the size of the matrix grows
import multiprocessing
import time
import numpy as np
#from matplotlib import pyplot as plt


mat_size = 3 #40320*2 #chosen to be 8! so the work can be divided up nicely for any number of cores from 1-8
mat1 = np.random.randint(10, size=(mat_size,mat_size))
mat2 = np.random.randint(10, size=(mat_size,mat_size))
mat3 = np.zeros((mat_size,mat_size), dtype = int)
time_list = [] #list for the total time taken for each no of cores

print(mat1)
print(mat2)

#the start&end parameters will tell the function what row to begin and finish on
def matrix_mult(first_row, last_row, mat_A, mat_B, mat_C):
    print('first row')
    print(first_row)
    print('last row')
    print(last_row)
    for i in range(first_row, last_row):
        for j in range(mat_size):
            print(' i')
            print(i)
            print(' j')
            print(j)
            for k in range(mat_size):
                #print('A')
                #print(mat_A[i][k])
                #print('B')
                #print(mat_B[k][j])
                mat_C[i,j] += (mat_A[i][k] * mat_B[k][j])
                #print(mat_C[i,j])
    print(mat_C)
    return mat_C


for i in range(mat_size):
    empty_res = np.zeros((mat_size,mat_size), dtype = int)
    mat3 = mat3 + matrix_mult(int(i * mat_size / 3), int((i + 1) * mat_size / 3), mat1, mat2, empty_res)

answer = np.matmul(mat1,mat2)
print('real answer')
print(answer)

print('my answer')
print(mat3)

comparison = mat3 == answer
equal_arrays = comparison.all()

if equal_arrays:
    print('correct')


#plt.plot([1,2,3,4,5,6,7,8],time_list)
#plt.show()

# !!! Put in something to check that its actually correct, like do it with numpy and check results

