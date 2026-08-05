#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid(float x) {
    float sig = 1 / (1 + __expf(-x));
    return sig;
}

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        Y[idx] = sigmoid(X[idx]);
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
