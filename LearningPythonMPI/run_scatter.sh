#!/bin/bash

for mat_size in 8 16 32 64 128 256; do
	for p in 1 2 4 8; do
		echo "$mat_size" >> scatter_results.txt
		echo "$p" >> scatter_results.txt
		/usr/bin/mpiexec -n $p python3 scatter.py $mat_size $mat_size >> scatter_results.txt
		echo "" >> scatter_results.txt
	done
done
