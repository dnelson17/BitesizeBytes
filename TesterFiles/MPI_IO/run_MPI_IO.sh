#!/bin/bash

python Gen_empty_dfs.py

for mat_size in 10 11 12 13 14 15 16; do
	echo "mat: 2^$mat_size"
	echo ""
	for i in 1 2 3 4 5 6 7 8 9 10; do
		echo "i: $i"
		mpiexec -n 1 python GenMatrices.py $mat_size $i
		for p in 1 2 4 8 16 32; do
			echo "p: $p"
			mpiexec -n $p python MatrixMult_MPI_IO.py $mat_size $i
			python Delete_C.py $mat_size $i
		done
		python Delete_A_B.py $mat_size $i
	done
done
