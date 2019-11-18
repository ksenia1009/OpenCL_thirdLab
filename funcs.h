#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <time.h>
#include <chrono>
#include <string>

#define BLOCK_SIZE 16

void printMatrix(float* matr, int n);
void printMatrix(float* matr, int m, int n);
void matrixMulti(float* a, float* b, float* c, int n);
void matrixMulti(float* a, float* b, float* c, int k, int m, int n);
void matrixMultiOMP(float* a, float* b, float* c, int n, int threadsNum);
void matrixMultiOMP(float* a, float* b, float* c, int k, int m, int n, int threadsNum);
bool checkResults(float* first, float* second, int n);
void task1_opencl(float* data_a, float* data_b, float* result, size_t SIZE, cl_ulong& start, cl_ulong& end, cl_device_type dtype);
void task2_opencl(float* data_a, float* data_b, float* result, size_t k, size_t m, size_t n, cl_ulong& start, cl_ulong& end, cl_device_type dtype);
void task3_opencl(float* data_a, float* data_b, float* result, size_t K, size_t M, size_t N, cl_ulong& start, cl_ulong& end, cl_device_type dtype);
