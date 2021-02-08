import numpy as np

f = open("mat_A.txt", "r") 

mat_size = 8
row_wanted = 3

#f.seek((25*mat_size+1)*(row_wanted+1))
f.seek((25*mat_size+1)*row_wanted)

mat_A = np.empty((mat_size*(3+1)))
print(mat_A)

print(f.read(25*mat_size+1))

print(mat_A)

  
# prints current postion 
#print(f.tell()) 
  
#print(f.readline())  
f.close() 
