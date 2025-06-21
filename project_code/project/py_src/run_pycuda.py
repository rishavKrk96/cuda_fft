import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
from pycuda  import gpuarray
import numpy
import numpy.linalg as la
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void add_kernel(int *a, int *b, int *c)  {
  int id = threadIdx.x;
  c[id] = a[id] + b[id];
}
""")

add = mod.get_function("add_kernel")

a = numpy.random.randint(1,20,5)
b = numpy.random.randint(1,20,5) 
c = numpy.zeros(5).astype(numpy.int32)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.empty(5, numpy.int32)

print(a_gpu.get())
print(b_gpu.get())

add(
        a_gpu, b_gpu, c_gpu, block=(5,1,1)
    )

print (c_gpu.get())
