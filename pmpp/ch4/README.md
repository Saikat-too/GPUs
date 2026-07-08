
## Architecture of a Modern GPU

A GPU is organized into an array of highly threaded streaming multiprocessors (SMs). Within each streaming multiprocessor are units called streaming processors or CUDA cores, which share control logic and memory resources. A GPU with more SMs will be able to run more blocks at the same time. SMs both execute computations and store the state available for computation in registers, with associated caches.

CUDA memory is the highest level of the memory hierarchy in the CUDA Programming model. It is stored in the GPU RAM. When a kernel is called, the CUDA runtime system launches a grid of threads that execute the kernel code.

Because there is a limited number of SMs and a maximum number of blocks that can be simultaneously assigned to each SM, there is a hard limit on the total number of blocks that can be simultaneously executing on a CUDA device.

### Calculating Blocks and Threads in a Grid

When processing an array of `N` elements, you must decide how many threads each block will contain (`BlockSize`). The CUDA Runtime then needs to know how many total blocks to launch (the `GridSize`).

* **Blocks in Grid:** `ceil(N / BlockSize)` (Typically coded as `(N + BlockSize - 1) / BlockSize`)
* **Total Threads in Grid:** `Blocks in Grid × BlockSize`

**Example:**
Imagine you have an array of **`N = 100`** elements, and you choose a **`BlockSize = 32`**.

* **Blocks:** We need `(100 + 32 - 1) / 32` = `131 / 32` = **`4 Blocks`**.
* **Total Threads Launched:** `4 Blocks × 32 Threads` = **`128 Threads`**.

Notice that we launched 128 threads to process 100 elements. This leads directly to the concept of inactive threads, which we will explore below.

---

## Synchronization

CUDA allows threads in the same block to coordinate their activities using a barrier synchronization function: `__syncthreads()`.

When a thread calls `__syncthreads()`, it will be held at the program location of the call until every thread in the same block reaches that location. All threads in a block must participate in the same barrier synchronization if a `__syncthreads()` statement is present.

```cpp
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
    int h_d[N];
    for (int i = 0; i < N; i++) h_d[i] = i;

    std::cout << "Original order of the Array : " << std::endl;
    for (int i = 0; i < N; i++) std::cout << h_d[i] << " ";
    std::cout << std::endl;

    int *d_d;
    cudaMalloc(&d_d, N * sizeof(int));
    cudaMemcpy(d_d, h_d, N * sizeof(int), cudaMemcpyHostToDevice);

    reverse<<<1, N>>>(d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Reverse order of the Array : " << std::endl;
    for (int i = 0; i < N; i++) std::cout << h_d[i] << " ";
    std::cout << std::endl;

    cudaFree(d_d);
    return 0;
}

```

---

## Warps and SIMD Hardware

Blocks can execute in any order relative to each other, which allows transparent scalability across different devices. A mobile processor may execute an application slowly but at extremely low power consumption, while a desktop processor may execute the same application at a higher speed while consuming more power. Thread scheduling in CUDA GPUs is a hardware-implemented concept.

For blocks that consist of multiple dimensions of threads, the dimensions will be projected into a linearized row-major layout before partitioning into warps. Imagine a two-dimensional block with 8×8 threads. These 64 threads will form two warps:

* The first warp starts from `T(0,0)` and ends with `T(3,7)`.
* The second warp starts with `T(4,0)` and ends with `T(7,7)`.

An SM is designed to execute all threads in a warp following the SIMD model (Single Instruction Multiple Data). At any instant in time, one instruction is fetched and executed for all threads in a warp.

### Active vs. Inactive Threads in a Warp

NVIDIA GPUs group threads strictly into 32-thread warps. Returning to our previous grid example of `BlockSize = 32` and `N = 100`:

* **Block 0 (Threads 0-31):** All 32 map to array elements. **Warp 0 is 100% Active.**
* **Block 1 (Threads 32-63):** All 32 map to array elements. **Warp 1 is 100% Active.**
* **Block 2 (Threads 64-95):** All 32 map to array elements. **Warp 2 is 100% Active.**
* **Block 3 (Threads 96-127):** Only threads 96, 97, 98, and 99 map to valid data.
* **Active Threads in Warp 3:** 4 threads.
* **Inactive Threads in Warp 3:** 28 threads.



Because the hardware SIMD unit executes in rigid blocks of 32, the GPU still spends cycles running those 28 inactive threads. We typically use an `if (idx < N)` guard in our code so those inactive threads simply bypass memory writes.

### Concealing SIMD and Control Divergence

The GPU programming model conceals SIMD operations by exposing each physical thread as a number of logical threads—as many logical threads as the SIMD width. Physical threads are called warps and the term "thread" is reserved for the logical thread. The length of the vector is called the SIMD width.

All instructions on NVIDIA GPUs are SIMD instructions (they operate on vectors, not scalars). After concealing the SIMD architecture of the GPU, the program is written as if it operates on scalars. However, SIMD hardware effectively restricts all threads in a warp to execute the exact same instruction at the same time.

For an `if-else` construct, execution works perfectly when either *all* threads in a warp execute the `if` path or *all* execute the `else` path. When threads within a warp take different control flow paths, the SIMD hardware must take multiple passes through these paths. We call this **control divergence** or **warp divergence**.

```cpp
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

```

### Analyzing Divergence in a Grid

Warp divergence occurs exclusively when threads **within the same warp** evaluate a branch condition differently.

**Example from the code above (`idx % 2 == 0`):**
In any given 32-thread warp (e.g., threads 0 to 31):

* 16 threads evaluate the condition as **True** (even threads).
* 16 threads evaluate the condition as **False** (odd threads).

**The Result:** The warp is completely divergent! The SIMD hardware cannot run the `if` and `else` blocks simultaneously. It masks out the odd threads, runs the `if` instructions for the even ones, then inverts the mask and runs the `else` instructions for the odd ones. This serialization cuts hardware efficiency in half.

**A Non-Divergent Example:**
What if your condition was `if (idx / 32 == 0)`?

* In **Warp 0** (threads 0-31), `idx / 32` is 0 for **all 32 threads**. They all take the `if` path. No divergence!
* In **Warp 1** (threads 32-63), `idx / 32` is 1 for **all 32 threads**. They all skip the `if` path. No divergence!
