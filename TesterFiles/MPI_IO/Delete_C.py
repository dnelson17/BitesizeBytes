import sys
import os

#Reads the Matrix Size from the command line
mat_power = int(sys.argv[1])
iteration = int(sys.argv[2])
mat_size = 2**mat_power

os.remove(f"mat_C/mat_C_{mat_size}_{iteration}.txt")
