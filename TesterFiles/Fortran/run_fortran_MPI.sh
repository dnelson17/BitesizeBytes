#!/bin/bash

source /etc/profile
mpiifort -mkl fortran_mpi.f90

for p in 1 2 4 8 16 32; do
	echo "p: $p"
	mpirun -n $p ./a.out >> fortran_results_no_lock.txt
	echo "" >> fortran_results_no_lock.txt
done

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpiifort -mkl fortran_mpi.f90

for p in 1 2 4 8 16 32; do
	echo "p: $p"
	mpirun -n $p ./a.out >> fortran_results_lock.txt
	echo "" >> fortran_results_lock.txt
done
