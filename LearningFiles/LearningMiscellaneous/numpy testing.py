import numpy as np

mat = np.zeros((4,4), dtype = int)
print(mat)

mat[2,2] = 8

mat[1,3] = 21

mat[3,1] = 5

print(mat)


x = 0
i = 0
while x < 50000:
    x += 840
    i += 1

print('x')
print(x)
print('i')
print(i)
