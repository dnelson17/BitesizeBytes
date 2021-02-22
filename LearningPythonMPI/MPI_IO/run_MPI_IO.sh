#!/bin/bash

for mat_size in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
	echo "$mat_size" >> mpi_io_results.txt
	echo "$mat_size"
	echo ""
	/usr/bin/mpiexec -n 1 python3 GenMatrices.py $mat_size $i
	for p in 1 2 4 8; do
		echo "$p" >> mpi_io_results.txt
		echo "$p"
		/usr/bin/mpiexec -n $p python3 MatrixMult_MPI_IO.py  $mat_size $i >> mpi_io_results.txt
	done
	echo ""
	echo "" >> mpi_io_results.txt
done
