#!/bin/bash

python Gen_empty_dfs.py

for mat_size in 8 9 10 11 12 13; do
	for i in 1 2 3 4 5 6 7 8 9 10; do
		echo "2^$mat_size"
		echo "i=$i"
		for p in 1 2 4 8 16 32; do
			echo "p=$p"
			mpiexec -n $p python MatrixMult_MPI_SGEMM.py $mat_size
			done
			echo ""
	done
done
