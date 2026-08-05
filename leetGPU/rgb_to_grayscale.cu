#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int total = height * width;

    if (idx < total) {
        int id = idx * 3;
        output[idx] = 0.299 * input[id] + 0.587 * input[id + 1] + 0.114 * input[id + 2];
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
