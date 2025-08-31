#include<stdio.h>
#include<cuda_runtime.h>

// Kernel Definition
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
