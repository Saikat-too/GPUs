# Multidimensional Grids and Data

In cuda al threads in a grid execute the same kernel function . **Example** :  Host code can be used to call the **vecAddKernel()** and generate a 1D grid that consists of **32** blocks , each of which consists **128** threads. Total number of threads in the grid is **32 * 128 = 4096**.

```
dim3 dimGrid(32,1,1);
dim3 dimBlock(128,1,1);
vecAddKernel<<<dimGrid , dimBlock>>>();
```

The default values of the parameters to the **dim3** constructors  are :
- **dimGrid** : (1 , 1 , 1)
- **dimBlock** : (1 , 1 , 1)

Within the kernel function , the **x** field of variables **gridDim** and **blockDim** are preinitialized according to the values of the execution cofiguration parameters . In **CUDA C** the allowed values of **grid.x** range from 1 to
**2^31 - 1**  and those of **gridDim.y** and **gridDim.z** range from 1 to **2^16 - 1**. All threads in a block share the same **blockIdx.x** , **blockIdx.y** and **blockIdx.z** values .

- One dimensional blocks can be created by setting **blockDim.y** and **blockDim.z** to **1**.
- Two dimensional blocks can be created by setting **blockDim.z** to **1**.

**N.B.** --> The total size of a block in current CUDA system is limited to **1024 threads**.


# Linearizing a 2D array

Two ways that we can linearized a 2D array are :
- **Row major layout**
- **Column major layout**

## Row major layout

It is to place all elements of the same row into consecutinve memory locations.

# Chapter 3: Matrix Multiplication in CUDA

This chapter demonstrates how to perform matrix multiplication using CUDA. The example code multiplies two square matrices, `M` and `N`, and stores the result in matrix `P`.

## Kernel Implementation

The `MatrixMultiplicationKernel` is a CUDA kernel that calculates the value for each element of the resulting matrix `P`. Each thread in the grid is responsible for calculating one element of `P`.

The thread indices `threadIdx.x` and `threadIdx.y` and block indices `blockIdx.x` and `blockIdx.y` are used to determine the row and column of the element that the current thread is responsible for.

```cuda
__global__ void MatrixMultiplicationKernel(float *M , float *N , float *P , int width)
{
   // Thread id
   int row = blockIdx.y * blockDim.y + threadIdx.y ;
   int col = blockIdx.x * blockDim.x + threadIdx.x  ;

   if((row < width) && (col < width)){
    float Pvalue = 0 ;
    for(int k = 0 ; k < width ; k++){
        Pvalue += M[row*width+k] * N[k*width+col];
     }
     P[row*width+col] = Pvalue ;
   }
}
```

## Host Implementation

The `main` function in the host code handles the following steps:

1.  **Initialization**: Initializes the input matrices `M` and `N` on the host.
2.  **Memory Allocation**: Allocates memory on the device for the matrices `Md`, `Nd`, and `Pd`.
3.  **Data Transfer**: Copies the input matrices `M` and `N` from the host to the device.
4.  **Execution Configuration**: Sets up the grid and block dimensions for the kernel launch.
5.  **Kernel Launch**: Launches the `MatrixMultiplicationKernel` on the device.
6.  **Result Transfer**: Copies the result matrix `P` from the device back to the host.
7.  **Cleanup**: Frees the allocated device memory.

```c
int main()
{
    // Initializing the arrays with width
    const int width = 5 ;
    float M[width*width] ;
    float N[width*width] ;
    float P[width * width];
    for(int i = 0 ; i < (width*width) ; i++){
        M[i] = 10 ;
        N[i] = 10 ;
        P[i] = 0 ;
    }


    // Declaring the variable and size for device .
    size_t size = width * width *sizeof(float);
    float *Md , *Nd , *Pd ;

    // Transfer M and N to device memory
    cudaMalloc((void**)&Md , size);
    cudaMemcpy(Md , M , size , cudaMemcpyHostToDevice);
    cudaMalloc((void**)&Nd , size);
    cudaMemcpy(Nd , N , size , cudaMemcpyHostToDevice);

    // Allocate the P on the device
    cudaMalloc((void**)&Pd , size);

    // Setup the execution configuration
    dim3 dimBlock(16 , 16);
    dim3 dimGrid((width + dimBlock.y - 1) /dimBlock.y ,  (width + dimBlock.x - 1)/dimBlock.x);


    // Launch the device computational thread
    MatrixMultiplicationKernel<<<dimGrid , dimBlock>>>(Md , Nd , Pd , width);

    // Check for errors

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("Kernel Launched failed: %s/n", cudaGetErrorString(err));
        return -1 ;

    }



    // Transfer P from device to host
    cudaMemcpy(P , Pd , size , cudaMemcpyDeviceToHost);

    // Print the output

    for( int i = 0 ;i < width ; i++){
        for(int j = 0 ; j< width ; j++){
            printf("%f " ,  P[i * width + j]);
        }
        printf("\n");
    }

    printf("The cuda operation is successful\n");
    //cleanup


    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);

    return 0 ;

}
```