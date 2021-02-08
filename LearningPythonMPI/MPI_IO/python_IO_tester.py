import numpy as np

def dist(size,mat_size):
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    send_list = []
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([[i*(mat_size/i_len),(i+1)*(mat_size/i_len)-1],[j*(mat_size/j_len),(j+1)*(mat_size/j_len)-1]])
    return send_list


def dist2(size,mat_size):
    power = np.log2(size)/2
    i_len = int(2**(np.ceil(power)))
    j_len = int(2**(np.floor(power)))
    i_size = mat_size/i_len
    j_size = mat_size/j_len
    send_list = []
    for i in range(i_len):
        for j in range(j_len):
            send_list.append([i*i_size,j*j_size])
    return send_list


f = open("mat_A.txt", "r") 

mat_size = 8
row_wanted = 3

#f.seek((25*mat_size+1)*(row_wanted+1))
#f.seek((25*mat_size+1)*row_wanted)

#mat_A = np.empty((mat_size*(3+1)))
#mat_A = np.empty(8)
#print(mat_A)

#mat_A = (f.read(25*mat_size+1)).split(" ")


#print(mat_A)

mat = np.loadtxt("mat_A.txt")
print(mat)

print("\n\n")

mat_A = np.loadtxt("mat_A.txt",skiprows=0,max_rows=4)
print(mat_A)

  
# prints current postion 
#print(f.tell()) 
  
#print(f.readline())


send_list = dist2(4,8)

print(send_list)






f.close() 
