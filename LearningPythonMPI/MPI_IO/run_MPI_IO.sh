#!/bin/bash

for mat_size in 128 256 512 1024 2048 4096 8192 16384 32768; do
	echo "$mat_size" >> mpi_io_results.txt
	echo "$mat_size"
	echo ""
	for i in 1 2 3 4 5 6 7 8 9 10; do
		/usr/bin/mpiexec -n 1 python3 GenMatrices.py $mat_size $i
		for p in 1 2 4 8; do
			echo "$p" >> mpi_io_results.txt
			echo "$p"
			/usr/bin/mpiexec -n $p python3 MatrixMult_MPI_IO.py  $mat_size $i >> mpi_io_results.txt
			python3 Delete_C.py $mat_size $i
		done
		python3 Delete_A_B.py $mat_size $i
	done
	echo ""
	echo "" >> mpi_io_results.txt
done
