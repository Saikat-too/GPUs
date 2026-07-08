#include<iostream>
#include<cuda_runtime.h>

const int N = 32;

__global__ void warp_divergence(int *d, int *even, int *odd){
    int idx = threadIdx.x;

    if (idx % 2 == 0) {
        even[idx / 2] = d[idx];
    }
    else {
        odd[idx / 2] = d[idx];
    }
}

int main() {
    // Host arrays
    int h_d[N];
    int h_even[N / 2];
    int h_odd[N / 2];

    for (int i = 0; i < N; i++) h_d[i] = i;

    std::cout << "Original array : " << std::endl;
    for (int i = 0; i < N; i++) std::cout << h_d[i] << " ";
    std::cout << std::endl;

    // Device pointers
    int *d_d, *d_even, *d_odd;

    cudaMalloc(&d_d, N * sizeof(int));
    cudaMalloc(&d_even, (N / 2) * sizeof(int));
    cudaMalloc(&d_odd, (N / 2) * sizeof(int));

    // Copy input from host to device
    cudaMemcpy(d_d, h_d, N * sizeof(int), cudaMemcpyHostToDevice);

    warp_divergence<<<1, N>>>(d_d, d_even, d_odd);
    cudaDeviceSynchronize();

    // Copy results back from device to host
    cudaMemcpy(h_even, d_even, (N / 2) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_odd, d_odd, (N / 2) * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Even-indexed elements : " << std::endl;
    for (int i = 0; i < N / 2; i++) std::cout << h_even[i] << " ";
    std::cout << std::endl;

    std::cout << "Odd-indexed elements : " << std::endl;
    for (int i = 0; i < N / 2; i++) std::cout << h_odd[i] << " ";
    std::cout << std::endl;

    cudaFree(d_d);
    cudaFree(d_even);
    cudaFree(d_odd);

    return 0;
}
