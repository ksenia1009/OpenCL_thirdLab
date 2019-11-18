#include "funcs.h"

const double threshold = 5e-10;

const char* matrixMultipleGPU = "__kernel void multi (             \n"
	"const int n,                                                  \n"
	"__global float* a,                                            \n"
	"__global float* b,                                            \n"
	"__global float* c                                             \n"
	") {                                                           \n"
	"int j = get_global_id(0);                                     \n"
	"int i = get_global_id(1);                                     \n"
	"float sum = 0.0;                                              \n"
	"for (int k = 0; k < n; k++) {                                 \n"
	"    sum += a[i * n + k] * b[k * n + j];                       \n"
	"}                                                             \n"
	"c[i * n + j] = sum;                                           \n"
	"}                                                             \n";

const char* matrixMultipleGPU_Optim =
"__kernel void multi_opt(const int m, const int n, const int k,    \n"
"	__global float* a,                                             \n"
"	__global float* b,                                             \n"
"	__global float* c) {                                           \n"
"	const int row = get_local_id(0);                               \n"
"	const int col = get_local_id(1);                               \n"
"	const int globalRow = BLOCK_SIZE * get_group_id(0) + row;      \n"
"	const int globalCol = BLOCK_SIZE * get_group_id(1) + col;      \n"
"	__local float Asub[BLOCK_SIZE][BLOCK_SIZE];                    \n"
"	__local float Bsub[BLOCK_SIZE][BLOCK_SIZE];                    \n"
"	float acc = 0.0f;                                              \n"
"	const int numTiles = n / BLOCK_SIZE;                           \n"
"	for (int t = 0; t < numTiles; t++) {                           \n"
"		const int tiledRow = BLOCK_SIZE * t + row;                 \n"
"		const int tiledCol = BLOCK_SIZE * t + col;                 \n"
"		Asub[col][row] = a[globalRow * n + tiledCol];              \n"
"		Bsub[col][row] = b[tiledRow * k + globalCol];              \n"
"		barrier(CLK_LOCAL_MEM_FENCE);                              \n"
"		for (int i = 0; i < BLOCK_SIZE; i++) {                     \n"
"			acc += Asub[i][row] * Bsub[col][i];                    \n"
"		}                                                          \n"
"		barrier(CLK_LOCAL_MEM_FENCE);                              \n"
"	}                                                              \n"
"	c[globalRow * k + globalCol] = acc;                            \n"
"}                                                                 \n";

const char* matrixMultiImg =
"kernel void multi_img(__read_only image2d_t a,                    \n"
"__read_only image2d_t b, __write_only image2d_t c) {              \n"
"  int row = get_local_id(0);                                      \n"
"  int col = get_local_id(1);                                      \n"
"  const int globalRow = BLOCK_SIZE * get_group_id(0) + row;       \n"
"  const int globalCol = BLOCK_SIZE * get_group_id(1) + col;       \n"
"  int n = get_global_size(0);                                     \n"
"  local float4 Asub[BLOCK_SIZE][BLOCK_SIZE];                      \n"
"  local float4 Bsub[BLOCK_SIZE][BLOCK_SIZE];                      \n"
"  float4 acc = 0.0f;                                              \n"
"  const int numTiles = n / BLOCK_SIZE;                            \n"
"  for (int t = 0; t < numTiles; t++) {                            \n"
"    const int tiledRow = BLOCK_SIZE * t + row;                    \n"
"    const int tiledCol = BLOCK_SIZE * t + col;                    \n"
"    const int2 idA = (tiledCol, globalRow);                       \n"
"    const int2 idB = (globalCol, tiledRow);                       \n"
"    Asub[col][row] = read_imagef(a, idA);                         \n"
"    Bsub[col][row] = read_imagef(b, idB);                         \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"    for (int i = 0; i < BLOCK_SIZE; i++) {                        \n"
"      acc += Asub[i][row] * Bsub[col][i];                         \n"
"     }                                                            \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"  }                                                               \n"
"  const int2 idC = (globalCol, globalRow);                        \n"
"  write_imagef(c, idC, acc);                                      \n"
"}                                                                 \n";

void printMatrix(float* matr, int n) 
{
	if (n < 10) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				std::cout << matr[i * n + j] << "\t";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}
void printMatrix(float* matr, int m, int n)
{
	if (n < 10) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				std::cout << matr[i * n + j] << "\t";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}
void matrixMulti(float* a, float* b, float* c, int n) 
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i * n + j] = 0;
			for (int l = 0; l < n; l++) {
				c[i * n + j] += a[i * n + l] * b[l * n + j];
			}
		}
	}
}
void matrixMulti(float* a, float* b, float* c, int k, int m, int n) 
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < k; j++) {
			c[i * k + j] = 0;
			for (int l = 0; l < n; l++) {
				c[i * k + j] += a[i * n + l] * b[l * k + j];
			}
		}
	}
}
void matrixMultiOMP(float* a, float* b, float* c, int n, int threadsNum)
{
	omp_set_num_threads(threadsNum);
	int i, j, l;
    #pragma omp parallel for private(i, j, l)
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			for (l = 0; l < n; l++) {
				c[i * n + j] += a[i * n + l] * b[l * n + j];
			}
		}
	}
}
void matrixMultiOMP(float* a, float* b, float* c, int k, int m, int n, int threadsNum) 
{
	omp_set_num_threads(threadsNum);
	int i, j, l;
    #pragma omp parallel for private(i, j, l)
	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			c[i * k + j] = 0;
			for (l = 0; l < n; l++) {
				c[i * k + j] += a[i * n + l] * b[l * k + j];
			}
		}
	}
}
bool checkResults(float* first, float* second, int n)
{
	bool check = false;
	for(int i = 0; i < n; i++)
	{
		if(fabs(first[i] - second[i]) <= threshold)
		{
			check = true;
		}
		else
		{
			check = false;
		}
	}
	return check;
}
void task1_opencl(float* data_a, float* data_b, float* result, size_t SIZE, cl_ulong& start, cl_ulong& end, cl_device_type dtype) 
{
	size_t n = SIZE;
	size_t group = 0;
	cl_int error = 0;

	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms); 
	cl_platform_id platform = NULL;

	if (0 < numPlatforms)
	{
		cl_platform_id* platforms = new cl_platform_id [numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];

		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
 		std::cout << "platform = " << platform_name << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties [3] = { 
		CL_CONTEXT_PLATFORM, ( cl_context_properties ) platform, 0 };

	cl_context context = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties,
		dtype,
		NULL,
		NULL,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	size_t size = 0;

	clGetContextInfo (
		context,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device = 0;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue = clCreateCommandQueue(
		context,		
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	size_t srclen[] = { strlen(matrixMultipleGPU) };

	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&matrixMultipleGPU,
		srclen,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program,
		1,
		&device,
		NULL,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build program failed: " << error << std::endl;
	}

	cl_kernel kernel = clCreateKernel(program,
		"multi",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	cl_mem a = clCreateBuffer (
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * SIZE * SIZE,
		NULL,
		NULL);

	cl_mem b = clCreateBuffer (
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * SIZE * SIZE,
		NULL,
		NULL);

	cl_mem c = clCreateBuffer (
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * SIZE * SIZE,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer (
		queue,
		a,
		CL_TRUE,
		0,
		sizeof(float) * SIZE * SIZE,
		data_a,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_a failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer (
		queue,
		b,
		CL_TRUE,
		0,
		sizeof(float) * SIZE * SIZE,
		data_b,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_b failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		0,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		1,
		sizeof(cl_mem),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		2,
		sizeof(cl_mem),
		&b);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		3,
		sizeof(cl_mem),
		&c);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for c failed: " << error << std::endl;
	}

	clGetKernelWorkGroupInfo (
		kernel,
		device,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof (size_t),
		&group,
		NULL );

	cl_event evt;
	auto start_gpu = std::chrono::steady_clock::now();

	size_t* global = new size_t[2];
	global[0] = (size_t)SIZE;
	global[1] = (size_t)SIZE;

	size_t* local = new size_t[2];
	local[0] = (size_t)BLOCK_SIZE;
	local[1] = (size_t)BLOCK_SIZE;

	int dim = 2;

	error = clEnqueueNDRangeKernel (
		queue,
		kernel,
		dim,
		NULL,
		global,
		local,
		0,
		NULL,
		&evt );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt);
	auto finish_gpu = std::chrono::steady_clock::now();

    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error << std::endl;
    }
    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error << std::endl;
    }

	clEnqueueReadBuffer (
		queue,
		c,
		CL_TRUE,
		0,
		sizeof(float) * n * n,
		result,
		0,
		NULL,
		NULL);

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
void task2_opencl(float* data_a, float* data_b, float* result, size_t K, size_t M, size_t N, cl_ulong& start, cl_ulong& end, cl_device_type dtype)
{
	size_t n = N, k = K, m = M;
	size_t group = 0;
	cl_int error = 0;

	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms); 
	cl_platform_id platform = NULL;

	if (0 < numPlatforms)
	{
		cl_platform_id* platforms = new cl_platform_id [numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];

		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
 		std::cout << "platform = " << platform_name << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties [3] = { 
		CL_CONTEXT_PLATFORM, ( cl_context_properties ) platform, 0 };

	cl_context context = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties,
		dtype,
		NULL,
		NULL,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	size_t size = 0;

	clGetContextInfo (
		context,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device = 0;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue = clCreateCommandQueue(
		context,		
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	std::string buildOpts = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);

	size_t srclen[] = { strlen(matrixMultipleGPU_Optim) };

	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&matrixMultipleGPU_Optim,
		srclen,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program,
		1,
		&device,
		buildOpts.c_str(),
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build prog failed" << std::endl;
		size_t logSize = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
		char *log = new char[logSize];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
		std::cout << log;
	}

	cl_kernel kernel = clCreateKernel(program,
		"multi_opt",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	cl_mem a = clCreateBuffer (
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * M * N,
		NULL,
		NULL);

	cl_mem b = clCreateBuffer (
		context,
		CL_MEM_READ_ONLY,
		sizeof(float) * N * K,
		NULL,
		NULL);

	cl_mem c = clCreateBuffer (
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * M * K,
		NULL,
		NULL);

	error = clEnqueueWriteBuffer (
		queue,
		a,
		CL_TRUE,
		0,
		sizeof(float) * M * N,
		data_a,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_a failed: " << error << std::endl;
	}

	error = clEnqueueWriteBuffer (
		queue,
		b,
		CL_TRUE,
		0,
		sizeof(float) * N * K,
		data_b,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_b failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		0,
		sizeof(int),
		&m);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for m failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		1,
		sizeof(int),
		&n);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for n failed: " << error << std::endl;
	}
	
	error = clSetKernelArg (
		kernel,
		2,
		sizeof(int),
		&k);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for k failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		3,
		sizeof(cl_mem),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		4,
		sizeof(cl_mem),
		&b);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		5,
		sizeof(cl_mem),
		&c);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for c failed: " << error << std::endl;
	}

	clGetKernelWorkGroupInfo (
		kernel,
		device,
		CL_KERNEL_WORK_GROUP_SIZE,
		sizeof (size_t),
		&group,
		NULL );

	cl_event evt;
	auto start_gpu = std::chrono::steady_clock::now();

	size_t* global = new size_t[2];
	global[0] = (size_t)M;
	global[1] = (size_t)K;

	size_t* local = new size_t[2];
	local[0] = (size_t)BLOCK_SIZE;
	local[1] = (size_t)BLOCK_SIZE;

	int dim = 2;

	error = clEnqueueNDRangeKernel (
		queue,
		kernel,
		dim,
		NULL,
		global,
		local,
		0,
		NULL,
		&evt );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt);
	auto finish_gpu = std::chrono::steady_clock::now();

    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error << std::endl;
    }
    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error << std::endl;
    }

	clEnqueueReadBuffer (
		queue,
		c,
		CL_TRUE,
		0,
		sizeof(float) * m * k,
		result,
		0,
		NULL,
		NULL);

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
void task3_opencl(float* data_a, float* data_b, float* result, size_t K, size_t M, size_t N, cl_ulong& start, cl_ulong& end, cl_device_type dtype)
{
	size_t n = N, k = K, m = M;
	size_t group = 0;
	cl_int error = 0;

	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms); 
	cl_platform_id platform = NULL;

	if (0 < numPlatforms)
	{
		cl_platform_id* platforms = new cl_platform_id [numPlatforms];
		clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[1];

		char platform_name[128];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platform_name, nullptr);
 		std::cout << "platform = " << platform_name << std::endl;

		delete[] platforms;
	}

	cl_context_properties properties [3] = { 
		CL_CONTEXT_PLATFORM, ( cl_context_properties ) platform, 0 };

	cl_context context = clCreateContextFromType (
		( NULL == platform ) ? NULL : properties,
		dtype,
		NULL,
		NULL,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create context from type failed: " << error << std::endl;
	}

	size_t size = 0;

	clGetContextInfo (
		context,
		CL_CONTEXT_DEVICES,
		0,
		NULL,
		&size );

	cl_device_id device = 0;

	if (size > 0)
	{
		cl_device_id * devices = ( cl_device_id * ) alloca ( size );
		clGetContextInfo (
			context,
			CL_CONTEXT_DEVICES,
			size,
			devices,
			NULL );
		device = devices[0];
		
		char device_name[128];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 128, device_name, nullptr);
		std::cout << device_name << std::endl;
	}

	cl_command_queue queue = clCreateCommandQueue(
		context,		
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create command queue with properties failed: " << error << std::endl;
	}

	std::string buildOpts = "-DBLOCK_SIZE=" + std::to_string(BLOCK_SIZE);

	size_t srclen[] = { strlen(matrixMultiImg) };

	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&matrixMultiImg,
		srclen,
		&error );

	if (error != CL_SUCCESS) {
		std::cout << "Create program failed: " << error << std::endl;
	}

	error = clBuildProgram(
		program,
		1,
		&device,
		buildOpts.c_str(),
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Build prog failed" << std::endl;
		size_t logSize = 0;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
		char *log = new char[logSize];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
		std::cout << log;
	}

	cl_kernel kernel = clCreateKernel(program,
		"multi_img",
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Create kernel failed: " << error << std::endl;
	}

	cl_image_format format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_R;

	cl_image_desc desc = {};

	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width =  N;
	desc.image_height = N;

	cl_mem a = clCreateImage (
		context,
		CL_MEM_READ_ONLY,
		&format,
		&desc,
		nullptr,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Set create image for a failed: " << error << std::endl;
	}

	cl_mem b = clCreateImage (
		context,
		CL_MEM_READ_ONLY,
		&format,
		&desc,
		nullptr,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Set create image for b failed: " << error << std::endl;
	}

	cl_mem c = clCreateImage (
		context,
		CL_MEM_WRITE_ONLY,
		&format,
		&desc,
		nullptr,
		&error);

	if (error != CL_SUCCESS) {
		std::cout << "Set create image for c failed: " << error << std::endl;
	}



	size_t origin[3] = {0, 0, 0};
    size_t region[3] = {N, N, 1};

	error = clEnqueueWriteImage (
		queue,
		a,
		CL_TRUE,
		origin,
		region,
		0,
		0,
		data_a,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_a failed: " << error << std::endl;
	}

	error = clEnqueueWriteImage (
		queue,
		b,
		CL_TRUE,
		origin,
		region,
		0,
		0,
		data_b,
		0,
		NULL,
		NULL);

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue write buffer data_b failed: " << error << std::endl;
	}



	error = clSetKernelArg (
		kernel,
		0,
		sizeof(cl_mem),
		&a);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for a failed: " << error << std::endl;
	}

	error = clSetKernelArg (
		kernel,
		1,
		sizeof(cl_mem),
		&b);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for b failed: " << error << std::endl;
	}
	
	error = clSetKernelArg (
		kernel,
		2,
		sizeof(cl_mem),
		&c);

	if (error != CL_SUCCESS) {
		std::cout << "Set kernel args for c failed: " << error << std::endl;
	}

	//clGetKernelWorkGroupInfo (
	//	kernel,
	//	device,
	//	CL_KERNEL_WORK_GROUP_SIZE,
	//	sizeof (size_t),
	//	&group,
	//	NULL );

	cl_event evt;
	auto start_gpu = std::chrono::steady_clock::now();

	size_t* global = new size_t[2];
	global[0] = (size_t)(N);
	global[1] = (size_t)(N);

	size_t* local = new size_t[2];
	local[0] = (size_t)BLOCK_SIZE;
	local[1] = (size_t)BLOCK_SIZE;

	int dim = 2;

	const size_t offsets[] = {0, 0};

	error = clEnqueueNDRangeKernel (
		queue,
		kernel,
		dim,
		offsets,
		global,
		local,
		0,
		NULL,
		&evt );

	if (error != CL_SUCCESS) {
		std::cout << "Enqueue failed: " << error << std::endl;
	}

	clWaitForEvents(1, &evt);
	auto finish_gpu = std::chrono::steady_clock::now();

	clEnqueueReadImage (
		queue,
		c,
		CL_TRUE,
		origin,
		region,
		0,
		0,
		result,
		0,
		NULL,
		NULL);

    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting start time: " << error << std::endl;
    }
    error = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    if (error != CL_SUCCESS) {
		std::cout << "Error getting end time: " << error << std::endl;
    }

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
