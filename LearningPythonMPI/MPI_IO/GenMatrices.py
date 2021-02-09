import numpy as np
import sys

mat_size = 8
# Will use this when testing such that it can be looped efficiently
#mat_size = int( sys.argv[1])
#iteration = int( sys.argv[2])

mat_A = np.random.rand(mat_size,mat_size)
#As B is an arbitrary matrix, we will take the transpose of it
mat_B = np.random.rand(mat_size,mat_size)
mat_B = np.transpose(mat_B)

print(mat_A)

np.savetxt('mat_A.txt', mat_A)
np.savetxt('mat_B.txt', mat_B)

A = np.loadtxt('mat_A.txt')
B = np.loadtxt('mat_B.txt')
print(mat_A)