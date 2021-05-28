def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[0]),dtype=np.float32)
    for i in range(mat_A.shape[0]):
        for j in range(mat_B.shape[0]):
            for k in range(mat_B.shape[1]):
                mat_C[i,j] += mat_A[i,k] * mat_B[j,k]
    return mat_C
