
#include<stdio.h>


#define N 1000

// Kernel Definition
__global__ void vectorAdd(double *a , double *b , double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

int main()
{
    // Number of bytes to allocate for N doubles

    size_t bytes  =  N * sizeof(double);

    // Allocate memory for arrays a , b and c on host
    double *a = (double*)malloc(bytes);
    double *b = (double*)malloc(bytes);
    double *c = (double*)malloc(bytes);


    // Alocate ojbect in device global memory
    double *a_d , *b_d , *c_d ;
    cudaMalloc(&a_d , bytes);
    cudaMalloc(&b_d , bytes);
    cudaMalloc(&c_d , bytes);

    // Fill the host arrays a nd b
    for (int i =  0 ; i < N ;i++){
        a[i] = 3.0 ;
        b[i] = 4.0 ;
    }

    // Memory data transfer from host to device
    cudaMemcpy(a_d , a , bytes , cuda   MemcpyHostToDevice);
    cudaMemcpy(b_d , b , bytes , cudaMemcpyHostToDevice);

    // Set execution configuration parmeters
    int thread_per_block = 256 ;
    int block_in_grid =  ceil(double(N) / thread_per_block) ;

    // Kernel execution
    vectorAdd<<<block_in_grid , thread_per_block>>>(a_d , b_d , c_d);

    // Copy data array from device array to host array
    cudaMemcpy(c , c_d , bytes , cudaMemcpyDeviceToHost);

    printf("The cuda operation is successful\n");
    printf("Thread per blocks = %d\n" , thread_per_block);
    printf("Block in Grid = %d\n" , block_in_grid);

    // Cpu  cleanup

    free(a);
    free(b);
    free(c);

    // Gpu cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0 ;

}
