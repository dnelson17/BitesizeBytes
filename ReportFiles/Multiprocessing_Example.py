from multiprocessing import Pool, shared_memory
import numpy as np

#Sgemm version of matrix multiplicaiton
def change_value(i,factor,shm_name):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(8, dtype=np.float32, buffer=existing_shm.buf)
    shared_array[i] *= factor
    existing_shm.close()


def main():
    #Generates an array of values [1,2,3,..8]
    data = np.arange(1,9)
    #Defines shared memory blocks array to be placed into
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    #Places array into shared memory
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    print(f"Array contents before operation: {shared_array}")
    #Gets the name of the sahred memory blocks so that they can be accessed by workers
    shm_name = shm.name
    #Creates a list of all parameters to be sent to workers. These paramaters include the "coordinates" of the matrices that should be work on and the name of the shared memory blocks that the matrices are sotred in.
    send_list = [[i,i+9,shm_name] for i in range(8)]
    #Opens a pool of worker processes
    p = Pool(processes=8)
    #Passes the parameters to each of the workers. Some timing results are then returned
    res_list = p.starmap(change_value, send_list)
    print(f"Array contents after operation: {shared_array}")
    #Closes the worker processes
    p.close()
    #Close and unlink the shared memory block
    shm.close()
    shm.unlink()


if __name__ == '__main__':
    main()
