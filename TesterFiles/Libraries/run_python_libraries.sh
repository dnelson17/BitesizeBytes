#!/bin/bash

python FunctionRawTimesComparison_without_naive.py 0

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python FunctionRawTimesComparison_without_naive.py 1
