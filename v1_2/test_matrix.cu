
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "matrix.h"


__global__ static void write(MATRIX* mat) {
    if(threadIdx.x < mat->row * mat->col) {
        MATRIX_write(mat, threadIdx.x/mat->col, threadIdx.x%mat->col, threadIdx.x+1);
    }
}

__global__ static void add(MATRIX* mat1, MATRIX* mat2) {
    if(threadIdx.x < mat1->row * mat1->col) {
        mat1->data[threadIdx.x/mat1->col][threadIdx.x%mat1->col] += mat2->data[threadIdx.x/mat1->col][threadIdx.x%mat1->col];
    }
}


int main()
{
    const int row = 7, col = 7;
    MATRIX *mat1_d, *mat2_d, *mat1_h, *mat2_h;

    mat1_d = MATRIX_new_device(row, col);
    mat2_d = MATRIX_new_device(row, col);
    
    write<<<1, 128, 0>>>(mat1_d);
    write<<<1, 128, 0>>>(mat2_d);

    MATRIX_move_to_host();

    // add<<<1, 128, 0>>>(mat1_d, mat2_d);

    mat1_h = MATRIX_create_the_same_on_host(mat1_d);
    mat2_h = MATRIX_create_the_same_on_host(mat2_d);

    printf("aaa\n");

    MATRIX_show(mat1_h, stdout);
    MATRIX_show(mat2_h, stdout);

    MATRIX_add(mat1_h, mat2_h);

    printf("aaa\n");

    MATRIX_show(mat1_h, stdout);

    printf("aaa\n");

    MATRIX_free(mat1_h);
    MATRIX_free(mat2_h);
    MATRIX_free_device(mat1_d);
    MATRIX_free_device(mat2_d);

    return 0;
}

