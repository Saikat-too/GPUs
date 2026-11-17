#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int total = width * height;
    unsigned char sub = 255;
    id = 3 * id + id;
    if (id < total * 4) {
        image[id] = sub - image[id];
        image[id + 1] = sub - image[id + 1];
        image[id + 2] = sub - image[id + 2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
