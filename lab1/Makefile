#nvcc = /opt/cuda-5.5/bin/nvcc
nvcc = nvcc

all: hello_cuda

hello_cuda: hello_cuda_skel.cu 
	$(nvcc) hello_cuda_skel.cu -o hello_cuda

clean:
	rm hello_cuda
