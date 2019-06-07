
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "tools.h"

#ifndef MATRIX_H_
#define MATRIX_H_


// 输入：矩阵、位置
// 功能：该位置的值加1
#define MATRIX_increase(mat, r, c) do{mat->data[r][c] += 1.;}while(0)

// 输入：矩阵、位置
// 功能：该位置的值减1
#define MATRIX_decrease(mat, r, c) do{mat->data[r][c] -= 1.;}while(0)


// 输入：矩阵、位置、值
// 功能：在矩阵的r行c列写入值v
#define MATRIX_write(mat, r, c, v) do{mat->data[r][c] = v;}while(0)

// 输入：矩阵、位置
// 功能：读取r行c列的值
#define MATRIX_read(mat, r, c) (mat->data[r][c])


// 16 bytes + 8*row + 4*row*col
typedef struct MATRIX {
    float** data;
    int row;
    int col;
} MATRIX;

// 输入：待创建矩阵的行数、列数
// 功能：创建一个row行col列的矩阵。并返回该矩阵的指针
MATRIX* MATRIX_new(int row, int col) {
    int i;
    MATRIX* mat;
    
    mat = (MATRIX*)malloc(sizeof(MATRIX));

    mat->data = (float**)malloc(sizeof(float*)*row);

    for(i=0; i<row; i++) {
        mat->data[i] = (float*)malloc(sizeof(float)*col);
        memset((void*)(mat->data[i]), 0, sizeof(float)*col);
    }

    mat->row = row;
    mat->col = col;

    return mat;
}

// 输入：矩阵指针
// 功能：释放矩阵占用的内存
void MATRIX_free(MATRIX* mat) {
    int i;

    // 读取矩阵的行和列
    for(i=0; i<mat->row; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}


// 在设备端创建一个row行col列的矩阵
MATRIX* MATRIX_new_device(int row, int col) {
    int i;
    MATRIX* mat_d;
    MATRIX mat_tmp;
    float* row_ptrs[row];

    // row & col
    mat_tmp.row = row;
    mat_tmp.col = col;

    // data
    cudaMalloc(&(mat_tmp.data), sizeof(float*)*row);
    for(i=0; i<row; i++) {
        cudaMalloc(&(row_ptrs[i]), sizeof(float)*col);
        cudaMemset(row_ptrs[i], 0, sizeof(float)*col);
    }
    cudaMemcpy(mat_tmp.data, row_ptrs, sizeof(float*)*row, cudaMemcpyHostToDevice);

    // 
    cudaMalloc(&mat_d, sizeof(MATRIX));
    cudaMemcpy(mat_d, &mat_tmp, sizeof(MATRIX), cudaMemcpyHostToDevice);
    
    return mat_d;
}

// mat指向显卡上的global memory
void MATRIX_free_device(MATRIX* mat_d) {
    int i;
    MATRIX mat_tmp;
    float** row_ptrs;

    // 拷贝到host端
    cudaMemcpy(&mat_tmp, mat_d, sizeof(MATRIX), cudaMemcpyDeviceToHost);
    cudaFree(mat_d);
    
    row_ptrs = (float**)malloc(sizeof(float*)*mat_tmp.row);
    cudaMemcpy(row_ptrs, mat_tmp.data, sizeof(float*)*mat_tmp.row, cudaMemcpyDeviceToHost);
    cudaFree(mat_tmp.data);
    
    for(i=0; i<mat_tmp.row; i++) {
        cudaFree(row_ptrs[i]);
    }

    free(row_ptrs);
}


// 将设备端的矩阵拷贝到主机端，两者的形状必须相同
void MATRIX_move_core_to_host(MATRIX* mat_h, float** core_d) {
    int i;
    float* row_ptrs[mat_h->row];

    cudaMemcpy(row_ptrs, core_d, sizeof(float*)*mat_h->row, cudaMemcpyDeviceToHost);
    for(i=0; i<mat_h->row; i++) {
        cudaMemcpy(mat_h->data[i], row_ptrs[i], sizeof(float)*mat_h->col, cudaMemcpyDeviceToHost);
    }
}

void MATRIX_free_core_device(float** core_d, int row) {
    int i;
    float* row_ptrs[row];

    cudaMemcpy(row_ptrs, core_d, sizeof(float*)*row, cudaMemcpyDeviceToHost);
    for(i=0; i<row; i++) {
        cudaFree(row_ptrs[i]);
    }
    cudaFree(core_d);
}

// 在设备端创造一个完全相同的矩阵
MATRIX* MATRIX_create_the_same_on_device(MATRIX* mat_h) {
    int i;
    MATRIX* mat_d;
    MATRIX mat_tmp;
    float** row_ptrs;

    mat_tmp.row = mat_h->row;
    mat_tmp.col = mat_h->col;

    row_ptrs = (float**)malloc(sizeof(float*)*mat_h->row);
    for(i=0; i<mat_h->row; i++) {
        cudaMalloc(&(row_ptrs[i]), sizeof(float)*mat_h->col);
        cudaMemcpy(row_ptrs[i], mat_h->data[i], sizeof(float)*mat_h->col, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&(mat_tmp.data), sizeof(float*)*mat_h->row);
    cudaMemcpy(mat_tmp.data, row_ptrs, sizeof(float*)*mat_h->row, cudaMemcpyHostToDevice);

    cudaMalloc(&mat_d, sizeof(MATRIX));
    cudaMemcpy(mat_d, &mat_tmp, sizeof(MATRIX), cudaMemcpyHostToDevice);

    free(row_ptrs);

    return mat_d;
}

// 在设备端创造矩阵的核
float** MATRIX_create_the_core_on_device(MATRIX* mat_h) {
    int i;
    float** core_d;
    MATRIX mat_tmp;
    float* row_ptrs[mat_h->row];

    for(i=0; i<mat_h->row; i++) {
        cudaMalloc(&(row_ptrs[i]), sizeof(float)*mat_h->col);
        cudaMemcpy(row_ptrs[i], mat_h->data[i], sizeof(float)*mat_h->col, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&core_d, sizeof(float*)*mat_h->row);
    cudaMemcpy(core_d, row_ptrs, sizeof(float*)*mat_h->row, cudaMemcpyHostToDevice);

    return mat_d;
}



// 在主机端创造一个完全相同的矩阵
MATRIX* MATRIX_create_the_same_on_host(MATRIX* mat_d) {
    int i;
    MATRIX *mat_h, mat_tmp;
    float** row_ptrs;

    cudaMemcpy(&mat_tmp, mat_d, sizeof(MATRIX), cudaMemcpyDeviceToHost);

    row_ptrs = (float**)malloc(sizeof(float*)*mat_tmp.row);
    cudaMemcpy(row_ptrs, mat_tmp.data, sizeof(float*)*mat_tmp.row, cudaMemcpyDeviceToHost);

    mat_h = MATRIX_new(mat_tmp.row, mat_tmp.col);

    for(i=0; i<mat_tmp.row; i++) {
        cudaMemcpy(mat_h->data[i], row_ptrs[i], sizeof(float)*mat_tmp.col, cudaMemcpyDeviceToHost);
    }

    return mat_h;
}


void MATRIX_show(MATRIX* mat, FILE* out) {
    int i, j;
    for(i=0; i<mat->row; i++) {
        for(j=0; j<mat->col; j++) {
            fprintf(out, "%f ", MATRIX_read(mat, i, j));
        }
        fputc('\n', out);
    }
}



// 输入：矩阵、列
// 功能：求该列的和
__host__ __device__ float MATRIX_col_sum(MATRIX* mat, int c) {
    int i;
    float result;

    result = 0.;    
    for(i=0; i<mat->row; i++) {
        result += mat->data[i][c];
    }

    return result;
}

// 输入：矩阵、行
// 功能：求该行的和
__host__ __device__ float MATRIX_row_sum(MATRIX* mat, int r) {
    int i;
    float result;

    result = 0.;
    for(i=0; i<mat->col; i++) {
        result += mat->data[r][i];
    }

    return result;
}

// 行归一化
__host__ __device__ void MATRIX_row_normalize(MATRIX* mat, int r) {
    int i;
    float sum;

    sum = 0.;
    for(i=0; i<mat->col; i++) {
        sum += mat->data[r][i];
    }
    for(i=0; i<mat->col; i++) {
        mat->data[r][i] /= sum;
    }
}

// 列归一化
__host__ __device__ void MATRIX_col_normalize(MATRIX* mat, int c) {
    int i;
    float sum;

    sum = 0.;
    for(i=0; i<mat->row; i++) {
        sum += mat->data[i][c];
    }
    for(i=0; i<mat->row; i++) {
        mat->data[i][c] /= sum;
    }
}

// mat的第r行从大到小排序
int cmp(const void * a, const void * b)
{
    return *(float*)b <= *(float*)a ? -1 : 1;
}
void MATRIX_row_sort(MATRIX* mat, int r) {
    qsort((void*)(mat->data[r]), mat->col, sizeof(float), cmp);
}


// 完整地创建一个MATRIX对象，和mat完全一样，并返回
MATRIX* MATRIX_copy(MATRIX* mat) {
    int i;
    MATRIX* res;

    res = (MATRIX*)malloc(sizeof(MATRIX));
    res = MATRIX_new(mat->row, mat->col);
    for(i=0; i<res->row; i++) {
        memcpy((void*)(res->data[i]), (void*)(mat->data[i]), res->col*sizeof(float));
    }
    return res;
}


// a <- a + b
__host__ __device__ void MATRIX_add(MATRIX* a, MATRIX* b) {
    int i, j;

    for(i=0; i<a->row; i++) {
        for(j=0; j<a->col; j++) {
            a->data[i][j] += b->data[i][j];
        }
    }
}

// a <- a - b
__host__ __device__ void MATRIX_sub(MATRIX* a, MATRIX* b) {
    int i, j;

    for(i=0; i<a->row; i++) {
        for(j=0; j<a->col; j++) {
            a->data[i][j] -= b->data[i][j];
        }
    }
}


// a <- b
__host__ __device__ void MATRIX_assign(MATRIX* a, MATRIX* b) {
    int i, j;
    for(i=0; i<a->row; i++) {
        for(j=0; j<a->col; j++) {
            a->data[i][j] = b->data[i][j];
        }
    }
}

// 检验两个矩阵是否相等
int MATRIX_equal(MATRIX* m1, MATRIX* m2) {
    int i, j;

    if(m1->row != m2->row || m1->col != m2->col) {
        return 0;
    }

    for(i=0; i<m1->row; i++) {
        for(j=0; j<m1->col; j++) {
            if(m1->data[i][j] != m2->data[i][j]) {
                return 0;
            }
        }
    }

    return 1;
}

float MATRIX_total_sum(MATRIX* mat) {
    int i, j;
    float sum;

    sum = 0.;
    for(i=0; i<mat->row; i++) {
        for(j=0; j<mat->col; j++) {
            sum += mat->data[i][j];
        }
    }

    return sum;
}

#endif