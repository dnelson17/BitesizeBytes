#!/bin/bash

python Gen_empty_dfs.py

for mat_size in 5 6 7 8 9 10 11 12 13; do
	echo "$mat_size"
	echo ""
	for i in 1 2 3 4 5 6 7 8 9 10; do
		echo "$i"
		for p in 1 2 4 8 16 32; do
			echo "$p"
			mpiexec -n $p python MatrixMult_MyFunc.py $mat_size
			done
			echo ""
	done
done
