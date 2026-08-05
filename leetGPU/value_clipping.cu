#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        if (input[idx] < lo) {
            output[idx] = lo;
        }
        else if (input[idx] > hi) {
            output[idx] = hi;
        }
        else output[idx] = input[idx];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
