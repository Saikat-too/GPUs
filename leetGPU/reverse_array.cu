#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N / 2) {
        float temp = input[idx];
        input[idx] = input[N-idx-1];
        input[N-idx-1] = temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
