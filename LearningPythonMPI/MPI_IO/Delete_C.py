import sys
import os

#Reads the Matrix Size from the command line
mat_size = int(sys.argv[1])
iteration = int(sys.argv[2])

os.remove(f"mat_C/mat_C_{mat_size}_{iteration}.txt")
