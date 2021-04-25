# Compilation

      mpiifort -mkl fortran_mpi.f90

# Execution

      mpirun -n 32 ./a.out

Here "32" is the number of cores being used.

# Output

The code outputs the timings for the range of matrix size (between 1024 and
32768) executed on a given number of cores. This is a little bit backwards but
it was the easiest way to implement it. It might be nice to plot the timings as
a function of matrix size, by the way, so you can show how the work scales up
with dimension and verify that it scales as you expect. 
