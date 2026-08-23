#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
     __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    float localMax = -INFINITY;
    float localNorm = 0.0f;

    for (unsigned int i = tid; i < N; i+=blockDim.x) {
        float x = input[i];
        if (x > localMax) {
            localNorm = localNorm * __expf(localMax - x);
            localMax = x;
        }

        localNorm += __expf(x - localMax);
    }
    __syncthreads();
    sdata[tid] = localMax;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0 ; s>>=1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float globalMax = sdata[0];
    __syncthreads();

    localNorm *= __expf(localMax - globalMax);
    sdata[tid] = localNorm;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s>>=1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float globalNorm = sdata[0];
    __syncthreads();

    for (unsigned int i = tid; i < N; i+=blockDim.x) {
        output[i] = __expf(input[i] - globalMax) / globalNorm;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
