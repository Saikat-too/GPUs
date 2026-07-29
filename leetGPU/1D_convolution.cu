#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
 int input_size, int kernel_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int n = input_size - kernel_size;
    if(idx <= n) {
        for (int j = 0 ; j < kernel_size; j++) {
            output[idx]+=input[idx+j] * kernel[j];
        }

    }
 }

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
