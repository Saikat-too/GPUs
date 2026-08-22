#include <cuda_runtime.h>

__global__ void reduction( const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float val = 0.0f;
    if (idx < N) val = input[idx];
    if (idx + blockDim.x < N) val += input[idx + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int elementsPerBlock = threadsPerBlock * 2;
    int blocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;

    cudaMemset(output, 0 , sizeof(float));
    reduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}
