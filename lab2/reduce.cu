#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>

using namespace std;

void checkResults ( int M, int N, int * output ) {
	for ( int i =0;i<M;i ++)
		if ( output [i] != N )
			printf("Err, idx %d = %d\n", i, output [i ]);
	memset ( output , 0, M*sizeof ( int ));
}

__global__ void reduce1(int* input, int* output, int N)
{
	int idx = threadIdx.x*N;
	int sum = 0;
	for (int i=0; i<N; i++)
		sum+=input[idx+i];
	output[threadIdx.x] = sum;
}

__global__ void reduce2(int* input, int* output, int N)
{
	int idx = threadIdx.x*N;
	int sum = 0;

#pragma unroll 32 //hint the compiler that we will read memory in next 32 iterations
	for (int i=0; i<N; i++)
		sum+=input[idx+i];

	output[threadIdx.x] = sum;
}

__global__ void reduce3(int* input, int* output, int N)
{
	int idx = (threadIdx.x + blockDim.x*blockIdx.x)*N;
	int sum = 0;

#pragma unroll 32 //hint the compiler that we will read memory in next 32 iterations
	for (int i=0; i<N; i++)
		sum+=input[idx+i];

	output[threadIdx.x + blockIdx.x * blockDim.x] = sum;
}

__global__ void reduce4(int * input, int * output, int N)
{
	extern __shared__ int shared[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	shared[threadIdx.x] = input[idx];
	__syncthreads();

	for(int i=1; i<blockDim.x; i*=2) {
		if ( threadIdx.x%(2*i) == 0)
			shared[threadIdx.x] += shared[threadIdx.x+i];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		output[blockIdx.x] = shared[0];
}

__global__ void reduce5(int * input, int * output, int N)
{
	extern __shared__ int shared[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	shared[threadIdx.x] = input[idx];
	__syncthreads();

	for (int i=1; i<blockDim.x; i*=2) {
		int index = 2*i*threadIdx.x;
		if (index < blockDim.x)
			shared[index] += shared[index+i];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		output[blockIdx.x] = shared[0];
}

__global__ void reduce6(int *input, int * output, int N)
{
	extern __shared__ int shared[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	shared[threadIdx.x] = input[idx];
	__syncthreads();

	for (int i=blockDim.x/2; i>0; i/=2) {
		if (threadIdx.x < i)
			shared[threadIdx.x] += shared[threadIdx.x+i];
		__syncthreads();
	}

	if (threadIdx.x == 0)
		output[blockIdx.x] = shared[0];
}

__global__ void reduce7(int *input, int * output, int N)
{
	extern __shared__ volatile int s[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	s[threadIdx.x] = input[idx];
	__syncthreads();

	if (threadIdx.x < 256)
		s[threadIdx.x] += s[threadIdx.x+256];
	__syncthreads();
	if (threadIdx.x < 128)
		s[threadIdx.x] += s[threadIdx.x+128];
	__syncthreads();
	if (threadIdx.x < 64)
		s[threadIdx.x] += s[threadIdx.x+64];
	__syncthreads();
	if (threadIdx.x < 32) {
		s[threadIdx.x] += s[threadIdx.x+32];
		s[threadIdx.x] += s[threadIdx.x+16];
		s[threadIdx.x] += s[threadIdx.x+8];
		s[threadIdx.x] += s[threadIdx.x+4];
		s[threadIdx.x] += s[threadIdx.x+2];
		s[threadIdx.x] += s[threadIdx.x+1];
	}
	__syncthreads();

	if (threadIdx.x == 0)
		output[blockIdx.x] = s[0];
}

int main () {
	const int N = 512 ; // num elems to reduce
	const int M = 256 ; // num arrays
	int *  input;
	int * devInput ;
	int *  output;
	int * devOutput ;
	cudaError_t status;

	cudaMalloc(&devInput , M*N* sizeof ( int )) ;
	cudaMalloc(&devOutput , M* sizeof (int )) ;
	cudaMallocHost(&input , M*N*sizeof ( int )) ;
	cudaMallocHost(&output , M*sizeof (int )) ;

	for ( int i =0;i<M*N;i++) {
		input[i] = 1;
	}
	// execute kernels

	status = cudaMemcpy(devInput, input, M*N*sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}
	reduce1<<<1, M>>>(devInput, devOutput, N);
	reduce2<<<1, M>>>(devInput, devOutput, N);
	reduce3<<<16, M/16>>>(devInput, devOutput,N);
	reduce4<<<M,N,N*sizeof(int)>>>(devInput,devOutput,N);
	reduce5<<<M,N,N*sizeof(int)>>>(devInput,devOutput,N);
	reduce6<<<M,N,N*sizeof(int)>>>(devInput,devOutput,N);
	reduce7<<<M,N,N*sizeof(int)>>>(devInput,devOutput,N);

	status = cudaMemcpy(output, devOutput, M*sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	checkResults(M,N,output);

	// end kernel execution
	cudaFree( devInput ) ;
	cudaFree( devOutput ) ;
	//delete[] input;
	//delete[] output;
	cudaFreeHost( input ) ;
	cudaFreeHost( output ) ;
	return 0;
}
