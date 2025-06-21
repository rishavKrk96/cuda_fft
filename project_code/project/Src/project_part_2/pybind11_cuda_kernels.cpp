/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern void  compute_dft_on_gpu(float* ip_real, float* ip_imaginary, float* op_real, float* op_imaginary , int size, int nx, int ny, int dft_dimx, int dft_dimy);
extern void  compute_fft_on_gpu(float* ip_real, float* ip_imaginary, float* op_real, float* op_imaginary, int N, int nx, int ny, int dimx, int dimy);

extern double time_spent_fft_host;
extern double time_spent_dft_host;
extern float kernel_time_fft;
extern float kernel_time_dft;

py::tuple dft_wrapper(py::array_t<float> py_ip_real, py::array_t<float> py_ip_imaginary, int nx, int ny , int dft_dimx, int dft_dimy)
{
	auto buf1 = py_ip_real.request();
	auto buf2 = py_ip_imaginary.request();

	// 1XM vector
	int N = py_ip_real.shape()[1];

	printf("N=%d\n", N);

	auto result_real = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one float item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{1,N},  /* Number of elements for each dimension */
		{ sizeof(float)*N, sizeof(float) }  /* Strides for each dimension */
	));

	auto result_imaginary = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one float item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{1,N },  /* Number of elements for each dimension */
		{ sizeof(float) * N, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result_real.request();
	auto buf4 = result_imaginary.request();

	float* ip_real = (float*)buf1.ptr;
	float* ip_imaginary = (float*)buf2.ptr;
	float* op_real = (float*)buf3.ptr;
	float* op_imaginary = (float*)buf4.ptr;


	int size = nx * ny;


	compute_dft_on_gpu(ip_real, ip_imaginary , op_real, op_imaginary, size, nx, ny, dft_dimx, dft_dimy);
	//cu_madd(A, B, C, M, N);

	py::tuple myTuple = py::make_tuple(result_real, result_imaginary, kernel_time_dft);

    return myTuple;
}


py::tuple fft_wrapper(py::array_t<float> py_ip_real, py::array_t<float> py_ip_imaginary, int nx, int ny, int dft_dimx, int dft_dimy)
{
	auto buf1 = py_ip_real.request();
	auto buf2 = py_ip_imaginary.request();

	// 1XM vector
	int N = py_ip_real.shape()[1];

	printf("N=%d\n", N);

	auto result_real = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one float item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ 1,N },  /* Number of elements for each dimension */
		{ sizeof(float) * N, sizeof(float) }  /* Strides for each dimension */
	));

	auto result_imaginary = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one float item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ 1,N },  /* Number of elements for each dimension */
		{ sizeof(float) * N, sizeof(float) }  /* Strides for each dimension */
	));

	auto buf3 = result_real.request();
	auto buf4 = result_imaginary.request();

	float* ip_real = (float*)buf1.ptr;
	float* ip_imaginary = (float*)buf2.ptr;
	float* op_real = (float*)buf3.ptr;
	float* op_imaginary = (float*)buf4.ptr;


	int size = nx * ny;


	compute_fft_on_gpu(ip_real, ip_imaginary, op_real, op_imaginary, size, nx, ny, dft_dimx, dft_dimy);
	//cu_madd(A, B, C, M, N);

	py::tuple myTuple = py::make_tuple(result_real, result_imaginary, kernel_time_fft);

	return myTuple;
}

PYBIND11_MODULE(project_part_2, m) {
    m.def("dft_wrapper", &dft_wrapper, "Compute DFT of the sequence");
	m.def("fft_wrapper", &fft_wrapper, "Compute FFT of the sequence");
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
