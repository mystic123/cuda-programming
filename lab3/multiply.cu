#include <cstdio>
#include <iostream>
#include <ctime>
#include <cuda_runtime_api.h>

using namespace std;

/* Matrix 4096 x 4096
	srand(1);
 */

const int N = 4096;
const int BLOCK = 16;

// Matrices are stored in row-major order: // M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	float* elements;
} Matrix;

__device__ inline float getElem(const Matrix M, int x, int y)
{
	return M.elements[x * N + y];
}

__device__ inline void setElem(Matrix M, int x, int y, float val)
{
	M.elements[x*N + y] = val;
}

__device__ inline Matrix getSub(Matrix M, int x, int y)
{
	Matrix sub;
	sub.elements = &M.elements[N*BLOCK * x + BLOCK*y];
	return sub;
}

//simple version
__global__ void multiply1(const Matrix A, const Matrix B, Matrix C)
{
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
	for (int e = 0; e < N; ++e)
		Cvalue += A.elements[row * N + e] * B.elements[e * N + col];
	C.elements[row * N + col] = Cvalue;
}

//with transponed matrix
__global__ void multiply2(const Matrix A, const Matrix B, Matrix C)
{
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
#pragma unroll
	for (int e = 0; e < N; ++e)
		Cvalue += A.elements[row * N + e] * B.elements[col * N + e];
	C.elements[row * N + col] = Cvalue;
}

__global__ void multiply4(Matrix A, Matrix B, Matrix C)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = N * BLOCK * by;
	int bBegin = BLOCK * by;

	int end   = aBegin + N - 1;

	int aStep = BLOCK;
	int bStep = BLOCK*N;

	float Csub = 0;

	for (int a = aBegin, b = bBegin; a <= end; a += aStep, b+=bStep) {

		//sub matrices
		__shared__ float As[BLOCK][BLOCK];
		__shared__ float Bs[BLOCK][BLOCK];

		//load values from global to shared mem
		As[ty][tx] = A.elements[a + N * ty + tx];
		Bs[ty][tx] = B.elements[b + N * ty + tx];

		__syncthreads();

#pragma unroll

		for (int k = 0; k < BLOCK; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		__syncthreads();
	}

	int c = N * BLOCK * by + BLOCK * bx;
	C.elements[c + N * ty + tx] = Csub;
}
//using shared memory
__global__ void multiply3(const Matrix A, const Matrix B, Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	Matrix Csub = getSub(C, blockRow, blockCol);

	float Cval = 0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (N/BLOCK); ++m) {
		Matrix Asub = getSub(A, blockRow, m);
		Matrix Bsub = getSub(B, m, blockCol);

		__shared__ float As[BLOCK][BLOCK];
		__shared__ float Bs[BLOCK][BLOCK];

		As[row][col] = getElem(Asub, row, col);
		Bs[row][col] = getElem(Bsub, row, col);

		__syncthreads();

		for (int e = 0; e < BLOCK; ++e) {
			Cval += As[row][e] * Bs[e][col];
		}

		__syncthreads();
	}

	setElem(Csub, row, col, Cval);
}

void transpose(Matrix M)
{
	for (int i = 0; i < N; i++) {
		for (int j = i+1; j < N; j++) {
			float tmp = M.elements[N*i + j];
			M.elements[N*i + j] = M.elements[N*j + i];
			M.elements[N*j + i] = tmp;
		}
	}
}

int main () {
	cudaError_t status;
	Matrix A,B,R,RT;
	Matrix devA,devB,devR;

	A.elements = new float[N*N];
	B.elements = new float[N*N];
	R.elements = new float[N*N];
	RT.elements = new float[N*N];
	//cudaMallocHost(&A.elements, N*N*sizeof(float)) ;
	//cudaMallocHost(&B.elements, N*N*sizeof(float)) ;
	//cudaMallocHost(&R.elements, N*N*sizeof(float)) ;
	//cudaMallocHost(&RT.elements, N*N*sizeof(float)) ;

	srand(time(0));
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			A.elements[N*i + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			B.elements[N*i + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}

	cudaMalloc(&devA.elements, N*N*sizeof(float)) ;
	cudaMalloc(&devB.elements, N*N*sizeof(float)) ;
	cudaMalloc(&devR.elements, N*N*sizeof(float)) ;

	status = cudaMemcpy(devA.elements, A.elements, N*N*sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	status = cudaMemcpy(devB.elements, B.elements, N*N*sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	dim3 threads(BLOCK, BLOCK);
	dim3 grid(N / threads.x, N / threads.y);
	multiply1<<<grid,threads>>>(devA,devB,devR);

	status = cudaMemcpy(R.elements, devR.elements, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	multiply3<<<grid, threads>>>(devA,devB,devR);

	status = cudaMemcpy(RT.elements, devR.elements, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	transpose(B);

	status = cudaMemcpy(devB.elements, B.elements, N*N*sizeof(float), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	multiply2<<<grid,threads>>>(devA,devB,devR);

	status = cudaMemcpy(RT.elements, devR.elements, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	cudaFree( devA.elements ) ;
	cudaFree( devB.elements ) ;
	cudaFree( devR.elements ) ;
	return 0;
}
