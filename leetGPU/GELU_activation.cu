#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* X, float* Y, int N) {
    int id = threadIdx.x + blockDim.x * blockIdx.x ;
    if (id >= N) return ;

    float root_two = 1.0f / sqrtf(2.0f);
    float divide_by_root_two = X[id] * root_two;
    float erf_val = erf(divide_by_root_two);
    float plus_one = 1.0f + erf_val;
    float result = X[id] * 0.5f * plus_one;

    Y[id] = result;
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
