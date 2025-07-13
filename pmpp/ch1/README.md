# Introduction

**CPU** is latency oriented . **GPU** design is throughput oriented . The speed of many **graphic** applications is limited by the rate data can be delivered from the memory system into processors and vice versa.  A **GPU** must be capable of moving extremely large amount of data into and out of graphics frame buffers in **DRAM** . The prevailing solution in **GPUs** is to optimize for the execution . The design style is commonly referred to as **throughput-oriented-design** , as it strives to maximize the total execution throughput of a large of threads while
allowing individual threads to take potentially much longer time to execute .

![gpu image](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)


The **GPU** is specialized for highly parallel computations and therefore designed such that more transistors are devoted to data processing rather than data caching and control flow .

In general, an application has a mix of parallel parts and sequential parts, so systems are designed with a mix of **GPUs and CPUs** in order to maximize overall performance. Applications with a high degree of parallelism can exploit this massively parallel nature of the GPU to achieve higher performance than on the CPU.
