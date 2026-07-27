#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
     int idx = threadIdx.x + blockDim.x * blockIdx.x;
     int total = width * height * 4;
     unsigned char sub = 255;
     if (idx < total && (idx & 3)!=3) {
        image[idx] = sub-image[idx];
     }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
