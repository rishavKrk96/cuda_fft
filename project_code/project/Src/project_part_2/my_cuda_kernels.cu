#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define M_PI 3.14159265358979323846264338327950288


double time_spent_fft_host;
double time_spent_dft_host;
float kernel_time_fft;
float kernel_time_dft;

__global__ void dft_gpu(float2* ip, float2* kernel, float2* op, int size, int nx)
{

    // Program kernel codes properly, otherwise your system could crash

    //2D thread block, 2D grid
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    int j;

    op[idx].x = 0;
    op[idx].y = 0;

    if (ix < size && iy < size) // As long as your code prevents access violation, you can modify this "if" condition.
    {

        for (j = 0; j < size; j++) {

            op[idx].x += kernel[j * size + idx].x * ip[j].x - kernel[j * size + idx].y * ip[j].y;
            op[idx].y += kernel[j * size + idx].x * ip[j].y + kernel[j * size + idx].y * ip[j].x;

        }


    }


}


//FFT GPU kernels
__global__ void fft_gpu(float2* ip, float2* op, int N, int m, int nx, int ny)
{
    //2D thread block, 2D grid
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    // N/M * M/2 matrix --> N/2 threads
    unsigned int row_num = idx / (m / 2);
    unsigned int col_num = idx % (m / 2);

    float omega_real = cosf(-2.0 * M_PI * (col_num) / m);
    float omega_imaginary = sinf(-2.0 * M_PI * (col_num) / m);

    float t_real = omega_real * ip[row_num * m + col_num + m / 2].x - omega_imaginary * ip[row_num * m + col_num + m / 2].y;
    float t_imaginary = omega_real * ip[row_num * m + col_num + m / 2].y + omega_imaginary * ip[row_num * m + col_num + m / 2].x;
    float u_real = ip[row_num * m + col_num].x;
    float u_imaginary = ip[row_num * m + col_num].y;

    op[row_num * m + col_num].x = u_real + t_real;
    op[row_num * m + col_num].y = u_imaginary + t_imaginary;
    op[row_num * m + col_num + m / 2].x = u_real - t_real;
    op[row_num * m + col_num + m / 2].y = u_imaginary - t_imaginary;

}


void initializeData(float2* ip, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i].x = (float)((rand() & 0xFF) - 0x80) / 0xFF;
        ip[i].y = (float)((rand() & 0xFF) - 0x80) / 0xFF;

        //ip_real[i] = cos(2.0 * M_PI * (size/2) * i/ size);
        //ip_imaginary[i] = sin(2.0 * M_PI * (size/2) * i/ size);

        /*if (i % 100 == 0) {
            printf("ip_real value at index %d = %f\n", i, ip_real[i]);
            printf("ip_imaginary value at index %d = %f\n", i, ip_imaginary[i]);
        }*/
    }



    return;
}

void initializeKernel(float2* kernel, int size)
{

    int i, j;

    for (i = 0; i < size; i++) {

        for (j = 0; j < size; j++) {

            kernel[i * size + j].x = cosf(-2.0 * M_PI * i * j / size);
            kernel[i * size + j].y = sinf(-2.0 * M_PI * i * j / size);

            /*if ((i % 100 == 0) && (j % 100 == 0)) {
                printf("kernel_real value at index %d = %f\n", i, kernel_real[i*size+j]);
                printf("kernel_imaginary value at index %d = %f\n", i, kernel_imaginary[i*size+j]);
            }*/

            //}

        }

    }


}


void compute_dft_on_host(float2* ip, float2* kernel, float2* op, int size)
{
    int i, j;

    for (i = 0; i < size; i++) {

        op[i].x = 0;
        op[i].y = 0;


        for (j = 0; j < size; j++) {

            op[i].x += kernel[i * size + j].x * ip[j].x - kernel[i * size + j].y * ip[j].y;
            op[i].y += kernel[i * size + j].x * ip[j].y + kernel[i * size + j].y * ip[j].x;

        }


    }

}


// Function to perform the non-recursive Cooley-Tukey FFT algorithm
void compute_fft_on_host(float2* ip, int N) {

    // Cooley-Tukey FFT
    for (int s = 1; s <= log2(N); s++) {
        int m = 1 << s;
        float omega_m_real = cos(-2.0 * M_PI / m);
        float omega_m_imaginary = sin(-2.0 * M_PI / m);
        for (int k = 0; k < N; k += m) {
            float omega_real = 1.0;
            float omega_imaginary = 0;
            float omega_real_prev = omega_real;
            float omega_imaginary_prev = omega_imaginary;
            for (int j = 0; j < m / 2; j++) {
                float t_real = omega_real * ip[k + j + m / 2].x - omega_imaginary * ip[k + j + m / 2].y;
                float t_imaginary = omega_real * ip[k + j + m / 2].y + omega_imaginary * ip[k + j + m / 2].x;
                float u_real = ip[k + j].x;
                float u_imaginary = ip[k + j].y;
                ip[k + j].x = u_real + t_real;
                ip[k + j].y = u_imaginary + t_imaginary;
                ip[k + j + m / 2].x = u_real - t_real;
                ip[k + j + m / 2].y = u_imaginary - t_imaginary;
                omega_real = (omega_real_prev * omega_m_real) - (omega_imaginary_prev * omega_m_imaginary);
                omega_imaginary = (omega_real_prev * omega_m_imaginary) + (omega_imaginary_prev * omega_m_real);
                omega_real_prev = omega_real;
                omega_imaginary_prev = omega_imaginary;


            }
        }
    }
}




void checkResult(float2* hostRef, float2* gpuRef, int N)
{
    double epsilon = 1.0E-0;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if ((abs(hostRef[i].x - gpuRef[i].x) > epsilon) || (abs(hostRef[i].y - gpuRef[i].y) > epsilon))
        {
            match = 0;
            printf("%d: host Real %f gpu Real %f  host Imaginary %f gpu Imaginary %f\n", i, hostRef[i].x, gpuRef[i].x, hostRef[i].y, gpuRef[i].y);
            break;
        }
        else {
            /*if (i % 100 == 0) {
                printf("host_ref value at index %d = %f\n", i, hostRef[i]);
                printf("gpu_ref value at index %d = %f\n", i, gpuRef[i]);
            }*/

        }
    }

    if (match)
        printf("PASS\n\n");
    else
        printf("FAIL\n\n");
}

void checkResult_dft_fft(float2* hostRef_dft, float2* hostRef_fft, int N)
{
    double epsilon = 1.0E-0;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if ((abs(hostRef_dft[i].x - hostRef_fft[i].x) > epsilon) || (abs(hostRef_dft[i].y - hostRef_fft[i].y) > epsilon))
        {
            match = 0;
            printf("%d: host_dft Real %f , host_fft Real %f  host_dft Imaginary %f , host_fft Imaginary %f\n", i, hostRef_dft[i].x, hostRef_fft[i].x, hostRef_dft[i].y, hostRef_fft[i].y);
            //break;
        }
        else {

            /*if (i % 100 == 0) {
                printf("host_ref DFT value at index %d = %f\n", i, hostRef_dft[i]);
                printf("host_ref FFT value at index %d = %f\n", i, hostRef_fft[i]);
            }*/
        }
    }

    if (match)
        printf("PASS\n\n");
    else
        printf("FAIL\n\n");
}


void compute_dft_on_gpu(float* h_ip_real, float* h_ip_imaginary, float* h_op_real, float* h_op_imaginary, int size, int nx, int ny, int dft_dimx, int dft_dimy) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // malloc host memory
    float2* hostRef_dft, * gpuRef_dft;

    hostRef_dft = (float2*)malloc(size * sizeof(float2));
    gpuRef_dft = (float2*)malloc(size * sizeof(float2));

    memset(hostRef_dft, 0, size * sizeof(float2));
    memset(gpuRef_dft, 0, size * sizeof(float2));

    // malloc host memory
    float2* h_ip,*h_kernel;
    h_ip = (float2*)malloc(size *  sizeof(float2));
    h_kernel = (float2*)malloc(size * size * sizeof(float2));
 
   

    // malloc device global memory
    float2* d_ip, *d_kernel,  *d_op;
    cudaMalloc((void**)&d_ip, size * sizeof(float2));
    cudaMalloc((void**)&d_kernel, size * size * sizeof(float2));
    cudaMalloc((void**)&d_op, size * sizeof(float2));

    //initialize DFT  kernel
    initializeKernel(h_kernel, size);

    //initialize Host data
    for (int i = 0; i < size; i++) {
        h_ip[i].x = h_ip_real[i];
        h_ip[i].y = h_ip_imaginary[i];
    }

    clock_t begin_dft = clock();

    //DFT on host
    compute_dft_on_host(h_ip, h_kernel, hostRef_dft, size);

    clock_t end_dft = clock();
    time_spent_dft_host = (double)(end_dft - begin_dft) / CLOCKS_PER_SEC;

    printf("DFT on host elapsed time: %lf ms\n", time_spent_dft_host * 1000);

    // 
    
    // transfer data from host to device
    cudaMemcpy(d_ip, h_ip, size * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, size * size * sizeof(float2), cudaMemcpyHostToDevice);


    // malloc device global memory
    float2* d_ip_fft, * d_op_fft;
    cudaMalloc((void**)&d_ip_fft, size * sizeof(float2));
    cudaMalloc((void**)&d_op_fft, size * sizeof(float2));

    // invoke kernel at host side


    dim3 block(dft_dimx, dft_dimy);
    dim3 grid((nx + block.x - 1) / (block.x), (ny + block.y - 1) / (block.y));

    cudaEventRecord(start, 0);

    //DFT on GPU
    dft_gpu << <grid, block >> > (d_ip, d_kernel, d_op, size, nx);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernel_time_dft, start, stop);

    // checkCudaErrors kernel error
    cudaGetLastError();

    // copy DFT kernel result back to host side
    cudaMemcpy(gpuRef_dft, d_op, size * sizeof(float2), cudaMemcpyDeviceToHost);

    checkResult(hostRef_dft, gpuRef_dft, size);


    printf("DFT on GPU <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
        grid.y,
        block.x, block.y, kernel_time_dft);


    for (int i = 0; i < size; i++) {
        h_op_real[i] = gpuRef_dft[i].x;
        h_op_imaginary[i] = gpuRef_dft[i].y;
    }


    // free device global memory
    cudaFree(d_ip);
    cudaFree(d_kernel);
    cudaFree(d_op);


    // free host memory
    free(h_ip);
    free(h_kernel);
    free(hostRef_dft);
    free(gpuRef_dft);

    // reset device
    cudaDeviceReset();

}




// Function to perform the non-recursive Cooley-Tukey FFT algorithm
void compute_fft_on_gpu(float* h_ip_real, float* h_ip_imaginary, float* h_op_real, float* h_op_imaginary, int size, int nx, int ny, int fft_dimx, int fft_dimy) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);


    float2* h_ip_fft;
    float2* hostRef_fft, * gpuRef_fft;

    h_ip_fft = (float2*)malloc(size * sizeof(float2));
    hostRef_fft = (float2*)malloc(size * sizeof(float2));
    gpuRef_fft = (float2*)malloc(size * sizeof(float2));

    memset(hostRef_fft, 0, size * sizeof(float2));
    memset(gpuRef_fft, 0, size * sizeof(float2));

    //initialize Host data
    for (int i = 0; i < size; i++) {
        h_ip_fft[i].x = h_ip_real[i];
        h_ip_fft[i].y = h_ip_imaginary[i];
    }


    // Bit-reversal permutation (optional, but helps with cache locality)
    for (int i = 0; i < size; i++) {
        int j = 0;
        int num_bits = floor(log2(size));
        for (int bit = 0; bit < num_bits; bit++) {
            j |= ((i >> bit) & 1) << (num_bits - 1 - bit);
        }
        if (j > i) {
            //complex double temp = a[i];
            float2 temp = h_ip_fft[i];
            h_ip_fft[i] = h_ip_fft[j];
            h_ip_fft[j] = temp;
        }
    }

    clock_t begin_fft = clock();

    //FFT on host
    compute_fft_on_host(h_ip_fft, size);

    clock_t end_fft = clock();
    time_spent_fft_host = (double)(end_fft - begin_fft) / CLOCKS_PER_SEC;

    printf("FFT on host elapsed time: %lf ms\n", time_spent_fft_host * 1000);

    for (int i = 0; i < size; i++) {

        hostRef_fft[i] = h_ip_fft[i];

    }

    //initialize Host data
    for (int i = 0; i < size; i++) {
        h_ip_fft[i].x = h_ip_real[i];
        h_ip_fft[i].y = h_ip_imaginary[i];
    }

    // malloc device global memory
    float2* d_ip_fft, * d_op_fft;
    cudaMalloc((void**)&d_ip_fft, size * sizeof(float2));
    cudaMalloc((void**)&d_op_fft, size * sizeof(float2));

    // Bit-reversal permutation (optional, but helps with cache locality)
    for (int i = 0; i < size; i++) {
        int j = 0;
        int num_bits = floor(log2(size));
        for (int bit = 0; bit < num_bits; bit++) {
            j |= ((i >> bit) & 1) << (num_bits - 1 - bit);
        }
        if (j > i) {
            //complex double temp = a[i];
            float2 temp = h_ip_fft[i];
            h_ip_fft[i] = h_ip_fft[j];
            h_ip_fft[j] = temp;

        }
    }

    // transfer data from host to device
    cudaMemcpy(d_ip_fft, h_ip_fft, size * sizeof(float2), cudaMemcpyHostToDevice);



    // invoke kernel at host side


    int s, m;

    int dimx = fft_dimx;
    int dimy = fft_dimy;

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / (block.x), (ny + block.y - 1) / (block.y * 2));

    cudaEventRecord(start, 0);

    // Cooley-Tukey FFT
    for (s = 1; s <= log2(size); s++) {
        m = 1 << s;
        if (s % 2 == 1) {
            fft_gpu << < grid, block >> > (d_ip_fft, d_op_fft, size, m, nx, ny);
        }
        else {
            fft_gpu << < grid, block >> > (d_op_fft, d_ip_fft, size, m, nx, ny);
        }

    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&kernel_time_fft, start, stop);

    // checkCudaErrors kernel error
    cudaGetLastError();

    int i = floor(log2(size));

    // copy DFT kernel result back to host side
    if (i % 2 == 0) {

        cudaMemcpy(gpuRef_fft, d_ip_fft, size * sizeof(float2), cudaMemcpyDeviceToHost);

    }
    else {

        cudaMemcpy(gpuRef_fft, d_op_fft, size * sizeof(float2), cudaMemcpyDeviceToHost);

    }

    checkResult(hostRef_fft, gpuRef_fft, size);


    printf("FFT on GPU <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
        grid.y,
        block.x, block.y, kernel_time_fft);


    for (int i = 0; i < size; i++) {
        h_op_real[i] = gpuRef_fft[i].x;
        h_op_imaginary[i] = gpuRef_fft[i].y;
    }


    // free device global memory
    cudaFree(d_ip_fft);
    cudaFree(d_op_fft);


    // free host memory
    free(h_ip_fft);
    free(hostRef_fft);
    free(gpuRef_fft);

    //printf("ao\n");

    // reset device
    cudaDeviceReset();
}