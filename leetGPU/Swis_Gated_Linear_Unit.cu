#include <cuda_runtime.h>

__device__ float SiLU(const float x){
    float silu = x * (1.0 / (1.0 + __expf(-x)));
    return silu;
}

__device__ float SWiGLU(const float x, const float silu){
    float swiglu = x * silu;
    return swiglu;
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < halfN) {
        float silu = SiLU(input[idx]);
        float swiglu = SWiGLU(input[idx + halfN], silu);
        output[idx] = swiglu;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
