from scipy.linalg import blas as FB
import numpy as np
import time

mat_size = 500

m1 = np.random.rand(mat_size,mat_size)
m2 = np.random.rand(mat_size,mat_size)

ans = np.matmul(m1,m2)

print("My Version")
m3 = np.zeros((mat_size,mat_size))
start = time.perf_counter()
for i in range(mat_size):
    for j in range(mat_size):
        for k in range(mat_size):
            m3[i,j] += m1[i][k] * m2[k][j]
finish = time.perf_counter()
time_taken = round(finish-start,8)
assert m3.all() == ans.all()
print(time_taken)
print("\n")

print("NumPy")
start = time.perf_counter()
mn = np.matmul(m1,m2)
finish = time.perf_counter()
time_taken = round(finish-start,8)
assert mn.all() == ans.all()
print(time_taken)
print("\n")

print("DGEMM")
start = time.perf_counter()
md = FB.dgemm(alpha=1, a=m1, b=m2)
finish = time.perf_counter()
time_taken = round(finish-start,8)
assert md.all() == ans.all()
print(time_taken)
print("\n")


print("SGEMM")
start = time.perf_counter()
ms = FB.sgemm(alpha=1, a=m1, b=m2)
finish = time.perf_counter()
time_taken = round(finish-start,8)
assert ms.all() == ans.all()
print(time_taken)
print("\n")
