import numpy as np

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[1]))
    #print(mat_A.shape[0])
    #print(mat_B.shape[1])
    for i in range(len(mat_A)):
        for j in range(len(mat_B[i])):
            for k in range(len(mat_B)):
                mat_C[i,j] += mat_A[i][k] * mat_B[k][j]
    return mat_C


def tester_1(size, mat_size):
    mat = np.random.rand(mat_size,mat_size)
    print(mat)
    print("")
    for i in range(size):
        first_i = round(i * mat_size / size)
        last_i = round((i + 1) * mat_size / size)
        for j in range(size):
            first_j = round(j * mat_size / size)
            last_j = round((j + 1) * mat_size / size)
            print(mat[first_i:last_i,first_j:last_j])


#this one works if p = 2**m for odd m
def tester_3(size, mat_size):
    #mat_A = np.random.rand(mat_size,mat_size)
    #mat_B = np.random.rand(mat_size,mat_size)
    mat_A = np.reshape(np.arange(mat_size**2),[mat_size,mat_size])
    mat_B = np.reshape(np.arange(mat_size**2,2*mat_size**2),[mat_size,mat_size])
    answer = np.matmul(mat_A,mat_B)
    print(mat_A)
    print(mat_B)
    print(answer)
    print("")
    #Assuming for simplicity that no processesors, size, is a power of 2
    p = int(np.sqrt(size))
    factor = 2**(int(np.log2(size))%2)
    #print(factor)
    send_list = []
    for i in range(factor*p):
        first_i = round(i * mat_size / (factor*p))
        last_i = round((i + 1) * mat_size / (factor*p))
        for j in range(p):
            first_j = round(j * mat_size / p)
            last_j = round((j + 1) * mat_size / p)
            #print(mat_A[first_i:last_i,:])
            #print(mat_B[:,first_j:last_j])
            mat_C = matrix_mult(mat_A[first_i:last_i,:],mat_B[:,first_j:last_j])
            #print(mat_C)
            print("\n")
            send_list.append([mat_A[first_i:last_i,:],mat_B[:,first_j:last_j]])
            print(f"i: {i}")
            print(f"j: {j}")
            print(f"Rank: {i*2 + j}")
    return send_list


#tester_2(4,8)
print("\n\n")
mat_list = tester_3(8,8)

#print("\n\n\n\n\n")

#for mat in mat_list:
#    mat_C = matrix_mult(mat[0],mat[1])
#    print(mat_C)

for i in range(10):
    n = 2**i
    print(f"n = 2^{i} = {2**i}")
    factor = int(2**(int(np.log2(n))%2))
    power = np.log2(n)/2
    print(f"power: {power}")
    print(np.arange(n).reshape(( int(2**np.ceil(power)) , int(2**np.floor(power)) )) )

