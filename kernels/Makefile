NVCC=nvcc -m64 --ptx

all: cuda_kernels.h

cuda_kernels.ptx: cuda_kernels.cu
	$(NVCC) --Wno-deprecated-gpu-targets -O3 -std c++17 -gencode arch=compute_50,code=compute_50 cuda_kernels.cu -o cuda_kernels.ptx

cuda_kernels.h: cuda_kernels.ptx
	bin2c cuda_kernels.ptx > cuda_kernels.h

clean:
	rm -f cuda_kernels.ptx cuda_kernels.h