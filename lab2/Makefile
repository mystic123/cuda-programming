#nvcc = /opt/cuda-5.5/bin/nvcc
nvcc = nvcc

all: redukcja 

hello_cuda: hello_cuda_skel.cu 
	$(nvcc) hello_cuda_skel.cu -o hello_cuda

redukcja: redukcja.cu
	$(nvcc) redukcja.cu -o redukcja

clean:
	rm hello_cuda
