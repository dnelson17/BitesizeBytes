#!/bin/bash

for mat_size in 8 16 32 64 128 256 512 1024 2048; do
	for p in 1 2 4 8; do
			echo "$mat_size" >> scatter_results.txt
			echo "$p" >> scatter_results.txt
			echo "$mat_size"
			echo "$p"
			echo ""
			for i in 1 2 3 4 5 6 7 8 9 10; do
				echo "$i"
				/usr/bin/mpiexec -n $p python3 scatter.py $mat_size $mat_size >> scatter_results.txt
			done
			echo ""
			echo "" >> scatter_results.txt
	done
done
