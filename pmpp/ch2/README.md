# Heterogenous Data Parallel Computing

## Data Parallelism


Inpput Array

| Column 1 | Column 2 | Column 3 | Column 4 | Column n |
|----------|----------|----------|----------|----------|
| I[0]  |  I[1]  | I[2]  | I[3]  | I[n]  |



Output Array

| Column 1 | Column 2 | Column 3 | Column 4 | Column n |
|----------|----------|----------|----------|----------|
| O[0]  |  O[1]  | O[2]  | O[3]  | O[n]  |


Data Parallelism in image to grayscale conversation pixels can be calculated independently.



## Structure


CUDA C program reflects the co-existence of a host(**CPU**) and one or more devices(**GPU**) in the computer.
- C ---> Host
- CUDA C ---> Host , Device

The execution starts with host code(**CPU Serial Code**) . When a kernel function is called a large number of threads are launched on the on a device to execute the kernel . All the threads that are launched by a kernel call are collectively called a grid . These threads are the primary vehicle of parallel execution in **CUDA platform** . When all threads of a grid have completed their execution , the grid terminates , and the execution continues on the host until another grid is launched .

In current **CUDA** systems , devices are often hardware cards that come with their own dynamic random access memory called device global memory .


## Kernel

Process

- Allocate space in global Memory .
- Host to Global memory .
- Decive exectution (result data) .
- Device global memory to **Host memory** .
- Free up allocated space in device **Global Memory** .


### cudaMalloc()
  - Allocate object in the device global memory .
  - Two parameters
     - Address
     - Size


### cudaFree
  - Free object from device global memory .
     - Pointer to free


### cudaMemcpy
  - Memory data transfer .
  - Requires Four parameters
     - Destination pointer
     - Source pointer
     - Number of bytes to copy
     - Direction


### Vector addition in CUDA
```CUDA

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
    cudaMemcpy(a_d , a , bytes , cudaMemcpyHostToDevice);
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
```

## Cuda Thread Basics

**1**. Cuda gives each thread a unique **ThreadID** to distinguish each other even though the kernel instructions are the same .


**2**.
```Cuda
  vectorAdd<<<block_in_grid , thread_per_block>>>(a_d , b_d , c_d)
```
here in the kernel call the memory arguments specify 4 blocks and 256 threads.

**3**.

 - Grids map to GPUs .
 - Blocks map to MultiProcessors .
 - Threads map to StreamProcessors .
 - Warps are groups of **(32)** threads that execute simultaneously .


![](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)



**4** .
Need to provide each kernel call with values two key structures .

  - Number of blocks in each dimension .
  - Number of threads per block in each dimension .



**5**.
The full global thread ID in x dimension can be computed by
```CUDA
  int id = blockDim.x * blockIdx.x + threadIdx.x
```



### Calculation


- 32 threads
- 4 blocks
- blockdDim.x = 8


For global thread ID 26 =>  blockIdx.x * blockDim.x + threadIdx.x = 3 * 8 + 2 = 26 .
