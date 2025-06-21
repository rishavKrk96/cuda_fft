import sys
#sys.path.append("../build/src/pybind11_cpp_examples/release")
sys.path.append("../build/Src/project_part_2/Debug")
import project_part_2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

##Perf measurements
N = 2**14
nx = 2**7
ny = 2**7
dimx_dft_list = [8,8,8,16,16,16,32,32,32,64,64,64,128,128,128]
dimx_fft_list = [4,4,4,8,8,8,16,16,16,32,32,32,64,64,64]
dimy_list = [2,4,8,2,4,8,2,4,8,2,4,8,2,4,8]

op_real_dft = np.zeros((1,N))
op_imaginary_dft = np.zeros((1,N))
op_real_fft = np.zeros((1,N))
op_imaginary_fft = np.zeros((1,N))

kernel_time_dft_list = np.zeros((15,2))
kernel_time_fft_list = np.zeros((15,30))

kernel_time_dft_avg = np.zeros((15,1))
kernel_time_fft_avg = np.zeros((15,1))

for m in range(0,15):
    for n in range(0,30):
        ip_real = np.random.randint(-0x80, high= 0x7F,  size=(1,N)) / 10
        ip_imaginary = np.random.randint(-0x80, high = 0x7F,size=(1,N)) / 10
        op_real_fft, op_imaginary_fft, kernel_time_fft = project_part_2.fft_wrapper(ip_real,ip_imaginary,nx,ny,dimx_fft_list[m],dimy_list[m])
        kernel_time_fft_list[m,n] = kernel_time_fft

kernel_time_fft_avg = np.mean(kernel_time_fft_list, axis = 1)

grid_x_dft_list = []
grid_x_fft_list = []
grid_y_list = []

for k in range(0,15):
    grid_x_fft_list.append(math.floor(nx / dimx_fft_list[k]))
    grid_x_dft_list.append(math.floor(nx / dimx_dft_list[k]))
    grid_y_list.append(math.floor(ny / dimy_list[k]))

for m in range(0,15):
    for n in range(0,2):
        ip_real = np.random.randint(-0x80, high= 0x7F,  size=(1,N)) / 10
        ip_imaginary = np.random.randint(-0x80, high = 0x7F,size=(1,N)) / 10
        op_real_dft, op_imaginary_dft, kernel_time_dft = project_part_2.dft_wrapper(ip_real,ip_imaginary,nx,ny,dimx_dft_list[m],dimy_list[m])
        #op_real_fft, op_imaginary_fft, kernel_time_fft = project_part_2.fft_wrapper(ip_real,ip_imaginary,nx,ny,dimx_list[m],dimy_list[m])
        kernel_time_dft_list[m,n] = kernel_time_dft
        #kernel_time_fft_list[m,n] = kernel_time_fft
    

kernel_time_dft_avg = np.mean(kernel_time_dft_list, axis = 1)


print("Average FFT Kernel run time for the following kernel configurations: ")
for k in range(0,15):
    new_str = " <<<(" + str(grid_x_fft_list[k]) + "," +  str(grid_y_list[k]) + "),(" + str(dimx_fft_list[k]) + "," + str(dimy_list[k]) + ")>>> Run Time: " + str(kernel_time_fft_avg[k])   
    print(new_str)


print("Average DFT Kernel run time for the following kernel configurations: ")
for k in range(0,15):
    new_str = " <<<(" + str(grid_x_dft_list[k]) + "," +  str(grid_y_list[k]) + "),(" + str(dimx_dft_list[k]) + "," + str(dimy_list[k]) + ")>>> Run Time: " + str(kernel_time_dft_avg[k])   
    print(new_str)



#Validation - FFT of a square wave
# Create an array of angles from 0 to 2*pi


N = 2**14
nx = 2**7
ny = 2**7
dimx = 32
dimy = 2


i = np.arange(0, N)

#print(N)
# Compute the sine values for the angles


ip_real = np.zeros((1,N))
ip_imaginary = np.zeros((1,N))

#ip_real[0,:] = np.cos((N/2) * 2 * np.pi * i / N)
#ip_imaginary[0,:] = np.sin((N/2) * 2 * np.pi * i / N)

ip_real[0,:] = np.sign(np.cos((100) * 2 * np.pi * i / N))

#print(i)
#print(ip_real[0,0:5])

op_real_fft = np.zeros((1,N))
op_imaginary_fft = np.zeros((1,N))
op_real_fft, op_imaginary_fft, kernel_time_fft = project_part_2.fft_wrapper(ip_real,ip_imaginary,nx,ny,dimx,dimy)
#print(op_real)

op_real_dft = np.zeros((1,N))
op_imaginary_dft = np.zeros((1,N))
op_real_dft, op_imaginary_dft, kernel_time_dft = project_part_2.dft_wrapper(ip_real,ip_imaginary,nx,ny,dimx,dimy)
#print(op_real)

op_abs_fft = np.sqrt(op_real_fft**2 + op_imaginary_fft**2)
op_abs_dft = np.sqrt(op_real_dft**2 + op_imaginary_dft**2)
xpoints = 2* np.pi * i/N - np.pi
fft_points = op_abs_fft[0,:].copy()
fft_points[0:int(N/2)] = op_abs_fft[0,int(N/2):N]
fft_points[int(N/2):N] = op_abs_fft[0,0:int(N/2)]
dft_points = op_abs_dft[0,:].copy()
dft_points[0:int(N/2)] = op_abs_dft[0,int(N/2):N]
dft_points[int(N/2):N] = op_abs_dft[0,0:int(N/2)]


ip_complex = ip_real + 1j * ip_imaginary
freqs = np.fft.fft(ip_complex)
ref_fft = np.abs(np.fft.fftshift(freqs)).reshape(-1)

print(np.abs(np.fft.fftshift(ip_complex)))

# Initialise the subplot function using number of rows and columns 
figure, axis = plt.subplots(1, 3) 
  
axis[0].plot(xpoints, fft_points) 
axis[0].set_title("CUDA FFT kernel") 
  
axis[1].plot(xpoints, dft_points) 
axis[1].set_title("CUDA DFT kernel") 
  
axis[2].plot(xpoints, ref_fft) 
axis[2].set_title("Numpy FFT Function") 

# Combine all the operations and display 
plt.show() 

