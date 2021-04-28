#!/bin/bash

source /etc/profile
mpiifort -mkl fortran_mpi.f90

for p in 1 2 4 8 16 32; do
	echo "p: $p"
	mpirun -n $p ./a.out >> fortran_results.txt
	echo "" >> fortran_results.txt
done
