#include<iostream>
#include<cuda_runtime.h>

const int N = 32;

__global__ void reverse(int *d){
    __shared__ int s[N];
    s[threadIdx.x] = d[threadIdx.x];
    __syncthreads();
    d[threadIdx.x] = s[N - threadIdx.x - 1];
}

int main() {
    // Host array
    int h_d[N];
    for (int i = 0; i < N; i++) h_d[i] = i;

    std::cout << "Original order of the Array : " << std::endl;
    for (int i = 0; i < N; i++) std::cout << h_d[i] << " ";
    std::cout << std::endl;

    // Device pointer
    int *d_d;
    cudaMalloc(&d_d, N * sizeof(int));

    // Copy input from host to device
    cudaMemcpy(d_d, h_d, N * sizeof(int), cudaMemcpyHostToDevice);

    reverse<<<1, N>>>(d_d);
    cudaDeviceSynchronize();

    // Copy result back from device to host
    cudaMemcpy(h_d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Reverse order of the Array : " << std::endl;
    for (int i = 0; i < N; i++) std::cout << h_d[i] << " ";
    std::cout << std::endl;

    cudaFree(d_d);

    return 0;
}
