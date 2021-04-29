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
    mat_A = np.arange(mat_size**2).reshape([mat_size,mat_size])
    mat_B = np.arange(mat_size**2,2*mat_size**2).reshape([mat_size,mat_size])
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
    return send_list, answer


#simplified version of tester 3 using numpy for you
def deconstruct(size,mat_size):
    #mat_A = np.random.rand(mat_size,mat_size)
    #mat_B = np.random.rand(mat_size,mat_size)
    mat_A = np.arange(mat_size**2).reshape([mat_size,mat_size])
    #mat_B = np.arange(mat_size**2,2*mat_size**2).reshape([mat_size,mat_size])
    mat_B = np.linalg.inv(mat_A)
    answer = np.matmul(mat_A,mat_B)
    print("mat A")
    print(mat_A)
    print("mat B")
    print(mat_B)
    print("answer")
    print(answer)
    print("")
    #want to do the splitting here, but use numpy to do it for you
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    send_list_A = np.split(mat_A, i_len, axis=0)
    send_list_B = np.split(mat_B, j_len, axis=1)
    #print(send_list_A)
    #print(send_list_B)
    send_list = []
    #print(len(send_list_A))
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([send_list_A[i],send_list_B[j]])

    return send_list, answer
    #print(send_list)


#simplified version of tester 3 using numpy for you
def deconstruct_with_np_repeat(size,mat_size):
    #mat_A = np.random.rand(mat_size,mat_size)
    #mat_B = np.random.rand(mat_size,mat_size)
    mat_A = np.arange(mat_size**2).reshape([mat_size,mat_size])
    mat_B = np.arange(mat_size**2,2*mat_size**2).reshape([mat_size,mat_size])
    answer = np.matmul(mat_A,mat_B)
    print(mat_A)
    print(mat_B)
    #print(answer)
    print("")
    #want to do the splitting here, but use numpy to do it for you
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    print("\n\n")
    print(np.split(mat_A, i_len, axis=0))
    print(np.split(mat_B, j_len, axis=1))
    print("")
    #print( np.concatenate( (np.split(mat_A, i_len, axis=0)) , axis=1) )
    #print("\n\n")
    send_list_A = np.tile( np.split(mat_A, i_len, axis=0), i_len)
    send_list_B = np.tile( np.split(mat_B, j_len, axis=1), j_len)
    #send_list_A = np.split( np.tile( np.split(mat_A, i_len, axis=0), i_len), i_len, axis=0)
    print(send_list_A)
    print(send_list_B)
    send_list_B = np.split(mat_B, j_len, axis=1)
    #send_list_A = np.repeat()
    

def reconstruct(size,mat_size,res_list):
    print("\n\n\n\n")
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    print(i_len)
    j_len = int(2**(np.floor(power)))
    print(j_len)
    
    res = np.zeros((size,size))

    print(res_list)
    x = np.concatenate(res_list,axis=1)
    print("x")
    print(x)
    y = np.split( x, i_len, axis=1)
    print("y")
    print(y)
    z = np.vstack( y )
    print("mine original (z)")
    print(z)
    #print("mine transpose (z^t)")
    #print(np.transpose(z))

    all_in_one = np.vstack( np.split( np.concatenate(res_list,axis=1) , i_len, axis=1) )
    print("all in one")
    print(all_in_one)
    return all_in_one

    

"""
#tester_2(4,8)
print("\n\n")
print("hello")
mat_list, answer = tester_3(4,4)

print(mat_list)
print("\n\n")

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

m0 = np.arange(4).reshape(2,2)
m1 = np.arange(4,8).reshape(2,2)
m2 = np.arange(8,12).reshape(2,2)
m3 = np.arange(12,16).reshape(2,2)

print(m0)
print(m1)
print(m2)
print(m3)

m = np.block([[m0,m1],[m2,m3]])

print(m)

a1 = np.concatenate((m0,m1,m2),axis=1)
print("a1")
print(a1)

a2 = np.concatenate((a1,m3),axis=1)
print("a2")
print(a2)

a3 = np.block([[a2],[a2]])
print("a3")
print(a3)

a4 = np.block([[a3],[a2]])
print("a4")
print(a4)

print("\n\n")


res_list = np.zeros((2,2))
print(res_list)
print("\n")

for pair in mat_list:
    print(pair[0])
    print(pair[1])
    res = matrix_mult(pair[0],pair[1])
    print(res)
    res_list = np.concatenate((res_list,res),axis=1)
    print(res_list)
    print("")

res_list = np.delete(res_list, (0,1,10,11))
res_list = np.reshape(res_list, (2,8))
print(res_list)
    
res_list = np.split(res_list, 2, axis=1)
print(res_list)

res_list = np.block([[res_list[0]],[res_list[1]]])
print(res_list)


print(answer)


print("\n\n\n\n\n\n\n")
"""

send_list, ans = deconstruct(8,8)

res_list = []
for mat in send_list:
    #res_list.append(np.matmul(mat[0],mat[1]))
    res_list.append(matrix_mult(mat[0],mat[1]))

print("res list")
print(res_list)

res = reconstruct(8,8,res_list)

print(np.array_equal(res, ans, equal_nan=False))

#mat_A = np.arange(8**2).reshape([8,8])
#mat_B = np.arange(8**2,2*8**2).reshape([8,8])

#print("real result")
#print(np.matmul(mat_A,mat_B))

