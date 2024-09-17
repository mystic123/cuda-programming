#nvcc = /opt/cuda-5.5/bin/nvcc
nvcc = nvcc

all: multiply

multiply: multiply.cu
	$(nvcc) -std=c++11 multiply.cu -o multiply

clean:
	rm multiply
