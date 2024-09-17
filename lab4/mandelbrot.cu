#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>

using namespace std;

const int N = 1024;
const int BX = 32;
const int BY = 8;
const double X0 = -2.0f;
const double X1 = 1.2f;
const double Y0 = -1.2f;
const double Y1 = 1.2f;
const int ITER = 1024;
const int ITERSTEP = 10;

//1 thread per block
__global__ void work(int *T)
{
	int row = blockIdx.x;

	double dx = (X1-X0)/(N-1);
	double dy = (Y1-Y0)/(N-1);

	double y = Y0 + row*dy;
	for (int i = 0; i < N; i++) {
		double x = X0 + i*dx;
		double Zx = x;
		double Zy = y;

		int k = 0;
		while ((k<ITER) && ((Zx*Zx + Zy*Zy)<4)) {
			double Zy2 = Zy;
			Zy = 2*Zx*Zy + y;
			Zx = Zx*Zx - Zy2*Zy2 + x;
			k++;
		}
		T[row*N + i] = k;
	}
}

//N threads per block
__global__ void work2(int *T)
{
	int row = blockIdx.x;

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double dx = (X1-X0)/(N-1);
	double dy = (Y1-Y0)/(N-1);

	double y = Y0 + row*dy;
	double x = X0 + threadIdx.x*dx;
	double Zx = x;
	double Zy = y;

	int k = 0;
	while ((k<ITER) && ((Zx*Zx + Zy*Zy)<4)) {
		double Zy2 = Zy;
		Zy = 2*Zx*Zy + y;
		Zx = Zx*Zx - Zy2*Zy2 + x;
		k++;
	}
	T[index] = k;
}

//dividing picture into blocks
__global__ void work3(int *T)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int index = row * N + blockIdx.x * blockDim.x + threadIdx.x;

	double dx = (X1-X0)/(N-1);
	double dy = (Y1-Y0)/(N-1);

	double y = Y0 + row*dy;
	double x = X0 + (threadIdx.x+blockIdx.x*blockDim.x)*dx;
	double Zx = x;
	double Zy = y;

	int k = 0;
	while ((k<ITER) && ((Zx*Zx + Zy*Zy)<4)) {
		double Zy2 = Zy;
		Zy = 2*Zx*Zy + y;
		Zx = Zx*Zx - Zy2*Zy2 + x;
		k++;
	}
	T[index] = k;
}

//check loop condition every ITERSTEP iterations
__global__ void work4(int *T)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int index = row * N + blockIdx.x * blockDim.x + threadIdx.x;

	double dx = (X1-X0)/(N-1);
	double dy = (Y1-Y0)/(N-1);

	double y = Y0 + row*dy;
	double x = X0 + (threadIdx.x+blockIdx.x*blockDim.x)*dx;
	double Zx = x;
	double Zy = y;

	int k = 0;
	while ((k<ITER) && ((Zx*Zx + Zy*Zy)<4)) {
#pragma unroll
		for (int k1 = 0; k1 < ITERSTEP; k1++) {
			double Zy2 = Zy;
			Zy = 2*Zx*Zy + y;
			Zx = Zx*Zx - Zy2*Zy2 + x;
			k++;
		}
	}
	T[index] = k;
}

//CPU version
void seq(int *T)
{
	double dx = (X1-X0)/(N-1);
	double dy = (Y1-Y0)/(N-1);

	for (int i = 0; i<N;i++) {
		double y = Y0 + i*dy;

		for (int j = 0; j<N;j++) {
			double x = X0 + j*dx;
			double Zx = x;
			double Zy = y;

			int k = 0;
			while (k<ITER && ((Zx*Zx+Zy*Zy)<4)) {
				double Zy2 = Zy;
				Zy = 2*Zx*Zy + y;
				Zx = Zx*Zx - Zy2*Zy2 + x;
				k++;
			}

			T[N*i + j] = k;
		}
	}
}

void makePictureInt(int *Mandel,int width, int height, int MAX){

	//double scale = 255.0/MAX;

	int red_value, green_value, blue_value;

	int MyPalette[35][3]={
		{255,0,255},
		{248,0,240},
		{240,0,224},
		{232,0,208},
		{224,0,192},
		{216,0,176},
		{208,0,160},
		{200,0,144},
		{192,0,128},
		{184,0,112},
		{176,0,96},
		{168,0,80},
		{160,0,64},
		{152,0,48},
		{144,0,32},
		{136,0,16},
		{128,0,0},
		{120,16,0},
		{112,32,0},
		{104,48,0},
		{96,64,0},
		{88,80,0},
		{80,96,0},
		{72,112,0},
		{64,128,0},
		{56,144,0},
		{48,160,0},
		{40,176,0},
		{32,192,0},
		{16,224,0},
		{8,240,0},
		{0,0,0}
	};

	FILE *f = fopen("Mandel.ppm", "wb");

	fprintf(f, "P3\n%i %i 255\n", width, height);
	//	printf("MAX = %d, scale %lf\n",MAX,scale);
	for (int j=0; j<height; j++) {
		for (int i=0; i<width; i++)
		{
			//if ( ((i%4)==0) && ((j%4)==0) ) printf("%d ",Mandel[j*width+i]);
			//red_value = (int) round(scale*(Mandel[j*width+i])/16);
			//green_value = (int) round(scale*(Mandel[j*width+i])/16);
			//blue_value = (int) round(scale*(Mandel[j*width+i])/16);
			int indx= (int) round(4*log2(Mandel[j*width+i]+1));
			red_value=MyPalette[indx][0];
			green_value=MyPalette[indx][2];
			blue_value=MyPalette[indx][1];

			fprintf(f,"%d ",red_value);   // 0 .. 255
			fprintf(f,"%d ",green_value); // 0 .. 255
			fprintf(f,"%d ",blue_value);  // 0 .. 255
		}
		fprintf(f,"\n");

	}
	fclose(f);
}

int main (int argc, char *argv[]) {
	if (argc < 2) {
		cout<<"Usage: 1 argument - implementation of algorithm:\n";
		cout<<"1 - CPU\n";
		cout<<"2 - GPU - 1 thread calucates 1 row of image\n";
		cout<<"3 - GPU - 1 thread calculates 1 pixel of image\n";
		cout<<"4 - GPU - with image divided into blocks\n";
		cout<<"5 - GPU - with image divided into blocks and checking loop condidtion with every "<<ITERSTEP<<" iterations\n";
		return 0;
	}
	int *T, *R;
	int *D;
	cudaError_t status;

	T = new int[N*N];
	R = new int[N*N];

	for (int i=0; i<N;i++) {
		for (int j=0; j<N; j++) {
			T[i*N+j] = 0;
		}
	}
	status = cudaMalloc(&D, N*N*sizeof(int)) ;

	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	dim3 threads(BX,BY);
	dim3 blocks(N/threads.x, N/threads.y);

	int s = atoi(argv[1]);
	switch (s) {
		case 1:
			seq(T);
			break;
		case 2:
			work<<<N,1>>>(D);
			break;
		case 3:
			work2<<<N,N>>>(D);
			break;
		case 4:
			work3<<<blocks,threads>>>(D);
			break;
		case 5:
			work4<<<blocks,threads>>>(D);
			break;
	}

	status = cudaMemcpy(R, D, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout<<cudaGetErrorString(status)<<endl;
	}

	makePictureInt(R, N, N, ITER);

	delete[] T;
	delete[] R;
	// Koniec kerneli
	cudaFree(&D) ;
	return 0;
}
