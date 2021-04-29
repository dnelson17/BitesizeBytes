import numpy as np

def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[1]))
    #print(mat_A.shape[0])
    #print(mat_B.shape[1])
    for i in range(len(mat_A)):
        for j in range(len(mat_B[i])):
            for k in range(len(mat_B)):
                mat_C[i,j] += mat_A[i][k] * mat_B[j][k]
    return mat_C


def deconstruct_with_np_repeat(size,mat_size):
    mat_A = np.arange(mat_size**2).reshape([mat_size,mat_size])
    mat_B = np.arange(mat_size**2,2*mat_size**2).reshape([mat_size,mat_size])
    mat_B = np.transpose(mat_B)
    print(mat_A)
    print(mat_B)
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    print("\n\n")
    print(np.split(mat_A, i_len, axis=0))
    print(np.split(mat_B, j_len, axis=0))
    print("\n\n")
    send_list_A = np.tile( np.split(mat_A, i_len, axis=0), i_len)
    send_list_B = np.tile( np.split(mat_B, j_len, axis=0), j_len)
    #send_list_A = np.split( np.tile( np.split(mat_A, i_len, axis=0), i_len), i_len, axis=0)
    print(send_list_A)
    print(send_list_B)

    send_list_A = np.vsplit(send_list_A, i_len)
    send_list_B = np.vsplit(send_list_B, j_len)
    
    print("\n\n")
    print(send_list_A)
    print(send_list_B)


deconstruct_with_np_repeat(4,4)
