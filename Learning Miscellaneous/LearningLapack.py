from scipy.linalg import blas as FB
import numpy as np
import time

mat_size = 20000

v1 = np.random.rand(mat_size,mat_size)
v2 = np.random.rand(mat_size,mat_size)

print("\n\n")

start = time.perf_counter()
vn = np.matmul(v1,v2)
finish = time.perf_counter()
time_taken = round(finish-start,4)
print(time_taken)
print("\n\n")

#start = time.perf_counter()
#vd = FB.dgemm(alpha=1, a=v1, b=v2)
#finish = time.perf_counter()
#time_taken = round(finish-start,4)
#print(time_taken)
#print("\n\n")

start = time.perf_counter()
vs = FB.sgemm(alpha=1, a=v1, b=v2)
finish = time.perf_counter()
time_taken = round(finish-start,4)
print(time_taken)
print("\n\n")
