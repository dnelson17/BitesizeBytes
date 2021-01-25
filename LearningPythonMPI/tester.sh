#!/bin/bash

for i in 16 32 64 128 256; do
	for j in 2 3 5 9; do
		echo "$i" >> results.txt
		echo "$j" >> results.txt
		/usr/bin/mpiexec -n $j python3 MatMul_tester.py $i $i >> results.txt
		echo "" >> results.txt
	done
done
