# Performance Analysis of DFT and FFT Implementations on GPUs 

## Author

Rishav Karki  

## Overview

This project analyzes and compares the performance of Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) implementations on CPU and GPU platforms using CUDA. The focus is on optimizing thread-level parallelism, memory access patterns, and execution time for N-point transforms.

## Contents

- Background: DFT and FFT
- DFT and FFT on CPU
- DFT and FFT on GPU with CUDA
- Validation criteria and methodology
- Performance results and analysis
- Conclusion

## DFT and FFT Algorithms

- **DFT** computes frequency components of time-domain signals and is commonly used in spectrum and signal analysis.
- **FFT** is a more efficient version of DFT, using the Cooley-Tukey algorithm, with a reduced time complexity of O(N log N).

## GPU Implementation Details

### CUDA DFT Kernel

- Uses a 2D thread block and grid.
- Each thread computes one output from N inputs, with a time complexity of O(N).
- Optimized memory access using symmetry and coalescing.

### CUDA FFT Kernel

- Uses a 2D thread block and grid.
- Executed log N times, each thread computes part of the butterfly operation.
- Efficient memory access with coalesced reads and no warp divergence.

## Validation Strategy

- Compared output of host DFT and FFT with their GPU counterparts.
- Used square wave inputs and visual inspection with Python (PyBind11).
- Measured execution time using CUDA events.
- Tested different thread block/grid sizes on NVIDIA RTX A2000 GPU.

## Results

### CPU Execution Time for N=214

- DFT: 1380 ms
- FFT: 1.0 ms

### GPU Execution Time for N=214

- DFT: 54.073 ms with thread block (64,2)
- FFT: 0.311 ms with thread block (8,8) and (16,4)

### Speedup

- GPU DFT vs CPU DFT: 25.52x
- GPU FFT vs CPU FFT: 3.21x

## Analysis

- Optimal performance is achieved when thread blocks are multiples of warp size (32 threads).
- Memory coalescing and aligned access reduce memory transaction overhead.
- Best performance observed with thread blocks (64,2) for DFT and (8,8)/(16,4) for FFT.

## How to Run

1. Extract `project.zip`
2. Open CMake and set source path: `C:/Users/TEMP/Downloads/project/project`
3. Build the project using Visual Studio
4. Run `run_pybind.py` from `C:/Users/TEMP/Downloads/project/project/py_src` in VS Code

## References

- Professional CUDA C Programming
- "A GPU Based Memory Optimized Parallel Method for FFT Implementation" – Fan Zhang et al.
- https://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html
