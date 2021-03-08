import numpy as np

mat_A = np.random.rand(4,4).astype(np.float32)
print("mat A")
print(mat_A)
mat_B = np.random.rand(4,4).astype(np.float32)
t_mat_B = np.transpose(mat_B)[:]
print("mat B (pre transpose)")
print(t_mat_B)
print("mat B (post transpose)")
print(mat_B)


print("mat c (1)")
print(np.matmul(mat_A,mat_B))
print("mat c (2)")
print(np.matmul(mat_A,t_mat_B))
