
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define thread_num 512

__device__ int sample_from_multinomial_on_device(float* prob, int n, long seed) {
    int i;
    float sum = 0., acm, u;
    curandState state;
    
    curand_init(seed, threadIdx.x, 0, &state);

    for(i=0; i<n; i++) {
        sum += prob[i];
    }
    
    // 生成U[0,1]的一个样本
    u = curand_uniform(&state);

    u = sum * u;

    acm = 0.;

    for(i=0; i<n; i++) {
        acm += prob[i];
        if(acm >= u) {
            return i;
        }
    }

    return n-1;
}

__global__ static void sample(int n, int* result, int seed) {
    curandState state;
    float prob[3] = {0.5, 0.3, 0.2};
    curand_init(seed, threadIdx.x, 0, &state);

    for(int i=threadIdx.x; i<n; i += thread_num) {
        result[i] = sample_from_multinomial_on_device(prob, 3, curand(&state));
        // result[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int arr_h[1000], *arr_d;
    srand((unsigned int)time(NULL));
    cudaSetDevice(0);
    cudaMalloc(&arr_d, sizeof(int)*1000);

    sample<<<1, thread_num, 0>>>(1000, arr_d, rand());
    cudaMemcpy(arr_h, arr_d, sizeof(int)*1000, cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++) {
        printf("%d\n", arr_h[i]);
    }

    cudaFree(arr_d);

    return 0;
}

/*
__global__ void generateRandom(long seed)
{
    curandState state;
    // int id = threadIdx.x;
    // long seed = rand;
    curand_init(seed, threadIdx.x, 0, &state);
    printf("random double: %f \n",abs(curand_uniform(&state)));
}



int main()
{
    srand((unsigned int)time(NULL));
    cudaSetDevice(0);
    generateRandom<<<1,16>>>(rand());
    cudaDeviceReset();
    return 0;
}
*/