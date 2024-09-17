#include <iostream>
#include <time.h>
#include <thread>
#include <cuda_runtime_api.h>

//#include <cutil.h>

using namespace std;

__global__ void kernel(int* array, int num_elements)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int step = gridDim.x * blockDim.x;

	for (; i < num_elements; i += step) {
		array[i] = 2 * array[i];
	}
}

int main()
{
	int* array_gpu;
	cudaError_t status;
	const int num_elements = 4096;
	int array_cpu[num_elements];

	status = cudaMalloc((void**)&array_gpu, sizeof(int)*num_elements);

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	for (int i=0; i<num_elements;i++) {
		array_cpu[i]=i;
	}

	status = cudaMemcpy(array_gpu, array_cpu, sizeof(int)*num_elements, cudaMemcpyHostToDevice);

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	kernel<<<2, 256, 0>>>(array_gpu, num_elements);
//	this_thread::sleep_for(chrono::seconds(3));
//	CUDA_SAFE_CALL_NO_SYNC(cudaThreadSynchronize();)
	status = cudaMemcpy(array_cpu, array_gpu, sizeof(int)*num_elements, cudaMemcpyDeviceToHost);

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	status=cudaFree(array_gpu);

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	for (int i=0; i<num_elements;i++) {
		cout<<array_cpu[i]<<" ";
	}
}
