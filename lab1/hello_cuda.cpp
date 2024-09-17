#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;

int main()
{
	int* array_gpu;
	cudaError_t status;
	const int num_elems = 4096;
	int array_cpu[num_elems];

	status = cudaMalloc((void**)&array_gpu, sizeof(int)*num_elems);

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	for (int i=0; i<num_elems;i++) {
		array_cpu[i]=i;
	}

	status=cudaFree(array_gpu);

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}
}
