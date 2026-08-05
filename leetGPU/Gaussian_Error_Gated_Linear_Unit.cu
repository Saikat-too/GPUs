#include <cuda_runtime.h>

__device__ float GELU(float x) {
    float gelu = 0.5 * x * (1.0 + (erf(x / sqrt(2))));
    return gelu;
}

__device__ float GEGLU(float x, float gelu) {
    float geglu = x * gelu;
    return geglu;
}

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < halfN) {
        float gelu = GELU(input[idx + halfN]);
        float geglu = GEGLU(input[idx], gelu);
        output[idx] = geglu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
