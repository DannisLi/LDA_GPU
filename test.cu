#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>



int main() {
    
    // int i, rows[6000], cols[1000000], vals[1000000];
    cudaDeviceProp prop;

    // printf("%d%d%d\n", rows[1], cols[2], vals[3]);

    if(cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("name: %s\n", prop.name);
        printf("Global Memory: %lu\n", prop.totalGlobalMem);
        printf("Shared Memory: %lu\n", prop.sharedMemPerBlock);
        printf("Registers: %d\n", prop.regsPerBlock);
        printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
        printf("totalConstMem: %lu\n", prop.totalConstMem);
    }


    return 0;
}

