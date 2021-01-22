#!/bin/bash

for i in 16 32 64 128 256; do
	/usr/bin/mpiexec -n 3 python3 MPI_Matmul.py $i $i >> my_file.txt
done

