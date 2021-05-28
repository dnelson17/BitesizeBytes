#In the "matrix_mult" handwritten verision of this, we only difference will be that the line that calls sgemm will instead call "matrix_mult" to perform tha matrix multiplication.

#So the difference is that this:
sub_mat_C = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B, trans_b=True)
#will change to this:
sub_mat_C = matrix_mult(sub_mat_A, sub_mat_B)

#Where this is the matrix_mult function:
def matrix_mult(mat_A, mat_B):
    mat_C = np.zeros((mat_A.shape[0],mat_B.shape[0]),dtype=np.float32)
    for i in range(mat_A.shape[0]):
        for j in range(mat_B.shape[0]):
            for k in range(mat_B.shape[1]):
                mat_C[i,j] += mat_A[i,k] * mat_B[j,k]
    return mat_C
