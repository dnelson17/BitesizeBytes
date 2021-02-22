#!/bin/bash

for mat_size in 8 16 32 64 128 256 512 1024 2048 4096 8192 16384; do
	echo "$mat_size" >> mpi_io_results.txt
	echo "$mat_size"
	for i in 1 2 3 4 5 6 7 8 9 10; do
		echo ""
		echo "$i"
		echo ""
		python3 GenMatrices.py $mat_size $i
		for p in 1 2 4 8; do
			echo "$p" >> mpi_io_results.txt
			echo "$p"
			/usr/bin/mpiexec -n $p python3 MatrixMultSGEMM.py  $mat_size $i >> mpi_io_results.txt
		done
		echo ""
		echo "" >> mpi_io_results.txt
	done
done
