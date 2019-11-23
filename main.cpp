#include "funcs.h"

const unsigned int SIZE = 256;
const int threadsNum = 4;

size_t K = 256;
size_t M = 256;
size_t N = 256;

int main()
{
	// --------------------------------------- TASK 1 ---------------------------------------

	std::cout << "---------- TASK 1 ----------" << std::endl;
	std::cout << "SIZE = " << SIZE << std::endl;
	std::cout << "Thread num = " << threadsNum << std::endl;

	size_t n = SIZE;
	size_t group = 0;

	float* data_a = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* data_b = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* result = (float*)_aligned_malloc(sizeof(float) * n * n, 64);

	float* data_a_cpu = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* data_b_cpu = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* result_cpu = (float*)_aligned_malloc(sizeof(float) * n * n, 64);

	float* data_a_omp = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* data_b_omp = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* result_omp = (float*)_aligned_malloc(sizeof(float) * n * n, 64);

	float* data_a_cpu_cl = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* data_b_cpu_cl = (float*)_aligned_malloc(sizeof(float) * n * n, 64);
	float* result_cpu_cl = (float*)_aligned_malloc(sizeof(float) * n * n, 64);

	for (int i = 0; i < n * n; i++) {
		float tmp_a = (float)rand() / 1000;
		data_a[i] = tmp_a;
		data_a_cpu[i] = tmp_a;
		data_a_omp[i] = tmp_a;
		data_a_cpu_cl[i] = tmp_a;
		float tmp_b = (float)rand() / 1000;
		data_b[i] = tmp_b;
		data_b_cpu[i] = tmp_b;
		data_b_omp[i] = tmp_b;
		data_b_cpu_cl[i] = tmp_b;

		result[i] = 0.0f;
		result_cpu[i] = 0.0f;
		result_omp[i] = 0.0f;
		result_cpu_cl[i] = 0.0f;
	}

	std::cout << "a:" << std::endl;
	printMatrix(data_a, n);
	std::cout << "b:" << std::endl;
	printMatrix(data_b, n);

	cl_ulong start = 0, end = 0;
	cl_ulong start_cpu_cl = 0, end_cpu_cl = 0; 

	task1_opencl(data_a, data_b, result, SIZE, start, end, CL_DEVICE_TYPE_GPU); //  OPENCL GPU

	clock_t start_cpu = clock();
	matrixMulti(data_a_cpu, data_b_cpu, result_cpu, n); //  CPU
	clock_t finish_cpu = clock();

	double start_omp = omp_get_wtime();
	matrixMultiOMP(data_a_omp, data_b_omp, result_omp, n, threadsNum); //  OPENMP
	double finish_omp = omp_get_wtime();

	task1_opencl(data_a_cpu_cl, data_b_cpu_cl, result_cpu_cl, SIZE, start_cpu_cl, end_cpu_cl, CL_DEVICE_TYPE_CPU); //  OPENCL CPU

	// ------------------------------------- Results -------------------------------------

	std::cout << std::endl;
	std::cout << "CPU RESULTS" << std::endl;
	printMatrix(result_cpu, n);
	std::cout << "OPENCL GPU RESULTS" << std::endl;
	printMatrix(result, n);
	std::cout << "OPENCL CPU RESULTS" << std::endl;
	printMatrix(result_cpu_cl, n);
	std::cout << "OPENMP RESULTS" << std::endl;
	printMatrix(result_omp, n);

	// -------------------------- Results comparison (CPU & GPU) --------------------------

	if(checkResults(result, result_cpu, n * n))
	{
		std::cout << "Results (CPU & GPU) are equal." << std::endl;
	}
	else
	{
		std::cout << "Results (CPU & GPU) are different." << std::endl;
	}

	// --------------------------------- Time ---------------------------------

	std::cout << "OPENCL GPU time = " << (cl_double)(end - start) * (cl_double)(1e-06) << " ms" << std::endl;  
	std::cout << "OPENCL CPU time = " << (cl_double)(end_cpu_cl - start_cpu_cl) * (cl_double)(1e-06) << " ms" << std::endl; 
	std::cout << "CPU time = " << (float)(finish_cpu - start_cpu) << " ms" << std::endl; 
	std::cout << "OMP time = " << (finish_omp - start_omp) * (1e+03) << " ms" << std::endl;

	// --------------------------------------- TASK 2 ---------------------------------------
	//
	// matrix A [ M * N ]
	// matrix B [ N * K ]
	// result: A * B = C [ M * K ]

	std::cout << "---------- TASK 2 ----------" << std::endl;
	std::cout << "K = " << K << " M = " << M << " N = " << N << std::endl;

	cl_ulong start2 = 0, end2 = 0;
	cl_ulong start_cpu_cl2 = 0, end_cpu_cl2 = 0;

	float* A = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_gpu = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_gpu = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_gpu = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_cpu = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_cpu = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_cpu = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_omp = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_omp = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_omp = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_img_gpu = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_img_gpu = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_img_gpu = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	float* A_img_cpu = (float*)_aligned_malloc(sizeof(float) * M * N, 64);
	float* B_img_cpu = (float*)_aligned_malloc(sizeof(float) * N * K, 64);
	float* C_img_cpu = (float*)_aligned_malloc(sizeof(float) * M * K, 64);

	for (int i = 0; i < M * N; i++) {
		float tmp_a = (float)rand() / 1000;
		A[i] = tmp_a;
		A_gpu[i] = tmp_a;
		A_cpu[i] = tmp_a;
		A_omp[i] = tmp_a;

		A_img_gpu[i] = tmp_a;
		A_img_cpu[i] = tmp_a;
	}
	for (int i = 0; i < N * K; i++) {
		float tmp_b = (float)rand() / 1000;
		B[i] = tmp_b;
		B_gpu[i] = tmp_b;
		B_cpu[i] = tmp_b;
		B_omp[i] = tmp_b;

		B_img_gpu[i] = tmp_b;
		B_img_cpu[i] = tmp_b;
	}
	for (int i = 0; i < M * K; i++) {
		C[i] = 0.0f;
		C_gpu[i] = 0.0f;
		C_cpu[i] = 0.0f;
		C_omp[i] = 0.0f;

		C_img_gpu[i] = 0.0f;
		C_img_cpu[i] = 0.0f;
	}

	std::cout << "A:" << std::endl;
	printMatrix(A, M, N);
	std::cout << "B:" << std::endl;
	printMatrix(B, N, K);

	clock_t start_cpu2 = clock();
	matrixMulti(A, B, C, K, M, N); //  CPU
	clock_t finish_cpu2 = clock();

	task2_opencl(A_gpu, B_gpu, C_gpu, K, M, N, start2, end2, CL_DEVICE_TYPE_GPU); //  OPENCL GPU

	task2_opencl(A_cpu, B_cpu, C_cpu, K, M, N, start_cpu_cl2, end_cpu_cl2, CL_DEVICE_TYPE_CPU); //  OPENCL CPU

	double start_omp2 = omp_get_wtime();
	matrixMultiOMP(A_omp, B_omp, C_omp, K, M, N, threadsNum); //  OPENMP
	double finish_omp2 = omp_get_wtime();

	// ------------------------------------- Results -------------------------------------

	std::cout << "CPU RESULTS" << std::endl;
	printMatrix(C, M, K);
	std::cout << "OPENCL GPU RESULTS" << std::endl;
	printMatrix(C_gpu, M, K);
	std::cout << "OPENCL CPU RESULTS" << std::endl;
	printMatrix(C_cpu, M, K);
	std::cout << "OPENMP RESULTS" << std::endl;
	printMatrix(C_omp, M, K);

	// -------------------------- Results comparison (CPU & GPU) --------------------------

	if(checkResults(C, C_gpu, M * K))
	{
		std::cout << "results (cpu & gpu) are equal." << std::endl;
	}
	else
	{
		std::cout << "results (cpu & gpu) are different." << std::endl;
	}

	// --------------------------------- Time ---------------------------------

	std::cout << "OPENCL GPU time = " << (cl_double)(end2 - start2) * (cl_double)(1e-06) << " ms" << std::endl; 
	std::cout << "OPENCL CPU time = " << (cl_double)(end_cpu_cl2 - start_cpu_cl2) * (cl_double)(1e-06) << " ms" << std::endl; 
	std::cout << "CPU time = " << (float)(finish_cpu2 - start_cpu2) << " ms" << std::endl; 
	std::cout << "OMP time = " << (finish_omp2 - start_omp2) * (1e+03) << " ms" << std::endl;

	// --------------------------------------- TASK 2 ---------------------------------------
	// Buffer ---> Image

	std::cout << "---------- TASK 3 ----------" << std::endl;

	cl_ulong start3 = 0, end3 = 0;

	task3_opencl(A_img_gpu, B_img_gpu, C_img_gpu, K, M, N, start3, end3, CL_DEVICE_TYPE_GPU);

	std::cout << "IMAGE OPENCL GPU RESULTS" << std::endl;
	printMatrix(C_img_gpu, M, K);

	// -------------------------- Results comparison (CPU & GPU) --------------------------

	if(checkResults(C_gpu, C_img_gpu, M * K))
	{
		std::cout << "Results (Image & GPU) are equal." << std::endl;
	}
	else
	{
		std::cout << "Results (Image & GPU) are different." << std::endl;
	}

	std::cout << "IMAGE OPENCL GPU time = " << (cl_double)(end3 - start3) * (cl_double)(1e-06) << " ms" << std::endl;

	_aligned_free(data_a);
	_aligned_free(data_a_cpu);
	_aligned_free(data_a_omp);
	_aligned_free(data_a_cpu_cl);
	_aligned_free(data_b);
	_aligned_free(data_b_cpu);
	_aligned_free(data_b_omp);
	_aligned_free(data_b_cpu_cl);
	_aligned_free(result);
	_aligned_free(result_cpu);
	_aligned_free(result_omp);
	_aligned_free(result_cpu_cl);

	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(A_gpu);
	_aligned_free(B_gpu);
	_aligned_free(C_gpu);
	_aligned_free(A_cpu);
	_aligned_free(B_cpu);
	_aligned_free(C_cpu);
	_aligned_free(A_omp);
	_aligned_free(B_omp);
	_aligned_free(C_omp);

	return 0;
}