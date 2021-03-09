from multiprocessing import Pool
from scipy.linalg import blas as FB
import numpy as np
import time

mat_sizes = [32,64,128,256,512,1024,2048]
no_runs = 10

#m1 = np.random.rand(mat_size,mat_size)
#m2 = np.random.rand(mat_size,mat_size)

#ans = np.matmul(m1,m2)

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[1]))
    for i in range(len(mat_A)):
        for j in range(len(mat_B[i])):
            for k in range(len(mat_B)):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    return mat_C


def gen_time_results(mat_size,no_cores):
    mat_A = np.random.rand(mat_size,mat_size)
    mat_B = np.random.rand(mat_size,mat_size)
    #Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
    power = np.log2(no_cores)/2
    #Represents the number of partitons that must be calculated in the result matrix C
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    #Represents the size of each partiton in the i and j axis
    i_size = int(mat_size/i_len)
    j_size = int(mat_size/j_len)
    if __name__ == '__main__':
        send_list = []
        for i in range(i_len):
            for j in range(j_len):
                send_list.append([mat_A[i*i_size:(i+1)*i_size,:],mat_B[:,j*j_size:(j+1)*j_size]])
        start = time.perf_counter()
        p = Pool(processes=no_cores)
        res_list = p.starmap(matrix_mult, send_list)
        p.close()
        finish = time.perf_counter()
        result = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
        time_taken = round(finish-start,10)
    return time_taken, result


print("----My Version - 1 Cores----")
for mat_size in mat_sizes[:-3]:#
    total_time_my_func_1 = 0
    print(f"Mat size: {mat_size}")
    for _ in range(no_runs):
        m1 = np.random.rand(mat_size,mat_size)
        m2 = np.random.rand(mat_size,mat_size)
        start = time.perf_counter()
        m3 = matrix_mult(m1,m2)
        finish = time.perf_counter()
        time_taken = round(finish-start,8)
        total_time_my_func_1 += time_taken
        #assert m3.all() == ans.all()
    print(total_time_my_func_1/no_runs)
print("\n")

"""
print("----My Version - 32 Cores----")
for mat_size in mat_sizes[:-3]:#
    total_time_my_func_32 = 0
    print(f"Mat size: {mat_size}")
    for _ in range(no_runs):
        time_taken, m3 = gen_time_results(mat_size,8)#Should be 32
        total_time_my_func_32 += time_taken
        #assert m3.all() == ans.all()
    print(total_time_my_func_32/no_runs)
print("\n")
"""

print("----NumPy----")
for mat_size in mat_sizes:
    print(f"Mat size: {mat_size}")
    total_time_Numpy = 0
    for _ in range(no_runs):
        m1 = np.random.rand(mat_size,mat_size)
        m2 = np.random.rand(mat_size,mat_size)
        start = time.perf_counter()
        mn = np.matmul(m1,m2)
        finish = time.perf_counter()
        time_taken = round(finish-start,8)
        total_time_Numpy += time_taken
        #assert mn.all() == ans.all()
    print(total_time_Numpy/no_runs)
print("\n")

print("----LAPACK DGEMM----")
for mat_size in mat_sizes:
    print(f"Mat size: {mat_size}")
    total_time_DGEMM = 0
    for _ in range(no_runs):
        m1 = np.random.rand(mat_size,mat_size)
        m2 = np.random.rand(mat_size,mat_size)
        start = time.perf_counter()
        md = FB.dgemm(alpha=1, a=m1, b=m2)
        finish = time.perf_counter()
        time_taken = round(finish-start,8)
        total_time_DGEMM += time_taken
        #assert md.all() == ans.all()
    print(total_time_DGEMM/no_runs)
print("\n")


print("---- LAPACK SGEMM----")
for mat_size in mat_sizes:
    print(f"Mat size: {mat_size}")
    total_time_SGEMM = 0
    for _ in range(no_runs):
        m1 = np.random.rand(mat_size,mat_size)
        m2 = np.random.rand(mat_size,mat_size)
        start = time.perf_counter()
        ms = FB.sgemm(alpha=1, a=m1, b=m2)
        finish = time.perf_counter()
        time_taken = round(finish-start,8)
        total_time_SGEMM += time_taken
        #assert ms.all() == ans.all()
    print(total_time_SGEMM/no_runs)
print("\n")
