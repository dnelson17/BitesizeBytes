from multiprocessing import Pool, shared_memory
from scipy.linalg import blas as FB
import pandas as pd
import numpy as np
import time

#Below are the parallel matrix multiplcation functions that will be called. In the real files, they will be in two separate files. The functions only vary in the part that is between the "-----", so I will only explain the first function

#Sgemm version of matrix multiplicaiton
def matrix_mult(i,len_i,j,len_j,mat_size,name_A,name_B,name_C):
    #The different cores often  have separate the timeer, so we mark the time before any calculation so that results can be offset to the same starting time
    stabiliser_time = time.time()
    #Identifies the shared memory blocks of A,B,C
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    existing_shm_C = shared_memory.SharedMemory(name=name_C)
    #Calculates the (i,j) coodinates of the submatrices that must be worked on
    i1 = i*len_i
    i2 = (i+1)*len_i
    j1 = j*len_j
    j2 = (j+1)*len_j
    #Reads the relevant block of A,B,C from shared memory
    sub_mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)[i1:i2,:]
    sub_mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)[:,j1:j2]
    sub_mat_C = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_C.buf)
    #Marks the start of the calculation time
    calc_start = time.time()
    #----------------------------------------------
    #Calculates the submatrix C' using sgemm and saves it to shared memory
    sub_mat_C[i1:i2,j1:j2] = FB.sgemm(alpha=1.0, a=sub_mat_A, b=sub_mat_B)
    #----------------------------------------------
    #<arks the end of the calculation time
    calc_finish = time.time()
    #Closes the link to the shared memory blocks
    existing_shm_A.close()
    existing_shm_B.close()
    existing_shm_C.close()
    #Returns all the timiing results
    return stabiliser_time, calc_start, calc_finish


#My handwritten version of matrix multiplicaiton
def matrix_mult(i,len_i,j,len_j,mat_size,name_A,name_B,name_C):
    stabiliser_time = time.time()
    existing_shm_A = shared_memory.SharedMemory(name=name_A)
    existing_shm_B = shared_memory.SharedMemory(name=name_B)
    existing_shm_C = shared_memory.SharedMemory(name=name_C)
    i1 = i*len_i
    i2 = (i+1)*len_i
    j1 = j*len_j
    j2 = (j+1)*len_j
    sub_mat_A = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_A.buf)
    sub_mat_B = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_B.buf)
    sub_mat_C = np.ndarray((mat_size,mat_size), dtype=np.float32, buffer=existing_shm_C.buf)
    calc_start = time.time()
    #----------------------------------------------
    #Calulates matrix C' using hadwritten matrix multiplication function
    for i in range(i1,i2):
        for j in range(j1,j2):
            for k in range(mat_size):
                sub_mat_C[i,j] += sub_mat_A[i,k] * sub_mat_B[k,j]
    #----------------------------------------------
    calc_finish = time.time()
    existing_shm_A.close()
    existing_shm_B.close()
    existing_shm_C.close()
    return stabiliser_time, calc_start, calc_finish


def gen_time_results(mat_size, core_list):
    #Generates 2 random matrices A,B
    data_A = np.random.rand((mat_size,mat_size)).astype(np.float32)
    data_B = np.random.rand((mat_size,mat_size)).astype(np.float32)
    #Generates empty matrix C for results to be written
    data_C = np.empty((mat_size,mat_size),dtype=np.float32)
    #Defines shared memory blocks for matrices to be place into
    shm_A = shared_memory.SharedMemory(create=True, size=data_A.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=data_B.nbytes)
    shm_C = shared_memory.SharedMemory(create=True, size=data_C.nbytes)
    #Places matrix A,B,C into shared memory
    mat_A = np.ndarray(data_A.shape, dtype=data_A.dtype, buffer=shm_A.buf)
    mat_A[:] = data_A[:]
    mat_B = np.ndarray(data_B.shape, dtype=data_B.dtype, buffer=shm_B.buf)
    mat_B[:] = data_B[:]
    mat_C = np.ndarray(data_C.shape, dtype=data_C.dtype, buffer=shm_C.buf)
    mat_C[:] = data_C[:]
    #Gets the name of the sahred memory blocks so that they can be accessed by workers
    name_A = shm_A.name
    name_B = shm_B.name
    name_C = shm_C.name
    #Define empty lists for various time results to be saved to
    total_times = []
    send_times = []
    calc_times = []
    recv_times = []
    #A list of cores to run on (typically [1,2,4,8,16,32]) will be passed from the main function. This loop will perform the matrix mutlplication on each of these processor counts.
    for no_cores in core_list:
        #Resets the values of matrix C to zeros after the previous matrix multiplcation
        mat_C[:] = np.zeros((mat_size,mat_size),dtype=np.float32)
        #Assuming the matrix is of size 2^n for int N, we take log2 to find the value of n
        power = np.log2(no_cores)/2
        #Represents the number of partitons that must be calculated in the result matrix C, phi_i and phi_j, respectively
        pars_i = int(2**(np.ceil(power)))
        pars_j = int(2**(np.floor(power)))
        #Represents the size of each partiton in the i and j axis, psi_i and psi_j, respectively
        len_i = int(mat_size/pars_i)
        len_j = int(mat_size/pars_j)
        #Starts overall timing
        total_start = time.time()
        #Creates a list of all parameters to be sent to workers. These paramaters include the "coordinates" of the matrices that should be work on and the name of the shared memory blocks that the matrices are sotred in.
        send_list = [[i,len_i,j,len_j,mat_size,name_A,name_B,name_C] for j in range(pars_j) for i in range(pars_i)]
        #Opens a pool of worker processes
        p = Pool(processes=no_cores)
        #Passes the parameters to each of the workers. Some timing results are then returned
        res_list = p.starmap(matrix_mult, send_list)
        #Closes the worker processes
        p.close()
        #Everything in the "----" below is just calculating timing results
        # ----------------------
        total_finish = time.time()
        calc_start_list = []
        calc_finish_list = []
        res_list = list(res_list)
        for i in range(len(res_list)):
            time_difference = res_list[0][0] - res_list[i][0]
            calc_start_list.append(res_list[i][1]+time_difference)
            calc_finish_list.append(res_list[i][2]+time_difference)
        calc_start = min(calc_start_list)
        calc_finish = max(calc_finish_list)
        send_time = calc_start-total_start
        calc_time = calc_finish-calc_start
        gather_time = total_finish-calc_finish
        total_time = total_finish-total_start
        assert send_time + calc_time + gather_time == total_time
        send_times.append( round(send_time,10) )
        calc_times.append( round(calc_time,10) )
        recv_times.append( round(gather_time,10) )
        total_times.append( round(total_time,10) )
        # ----------------------
    #Closes and unlinks the shared memory memory blocks that were created for A,B,C
    shm_A.close()
    shm_B.close()
    shm_C.close()
    shm_A.unlink()
    shm_B.unlink()
    shm_C.unlink()
    #Returns timing results to be saved to dataframes
    return tuple(send_times), tuple(calc_times), tuple(recv_times), tuple(total_times)
