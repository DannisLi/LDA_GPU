
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#ifndef TOOLS_H_
#define TOOLS_H_

// 是否处于调试状态
#define DEBUG 0

// 是否使用GPU
// #define USE_GPU 0

// CUDA grid配置
#define block_num 1
#define thread_num 256

// 最大单词表长度
#define MAX_VOC_SIZE 30000
// 最大文档数量
#define MAX_DOC_NUM 40000
// 最大主题数量
#define MAX_TOPIC_NUM 20

#define max(A, B) ((A) > (B) ? (A) : (B))
#define min(A, B) ((A) < (B) ? (A) : (B))

// 语料库路径
#define MY_COR "../data/my.corpus"
#define KOS_COR "../data/kos.corpus"
#define NIPS_COR "../data/nips.corpus"
#define ENRON_COR "../data/enron.corpus"

// 单词表路径
#define MY_VOC "../data/my.vocab"
#define KOS_VOC "../data/kos.vocab"
#define NIPS_VOC "../data/nips.vocab"
#define ENRON_VOC "../data/enron.vocab"


bool InitCUDA()
{
    int count;

    // 读取设备数量
    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    bool flag = false;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                flag = true;
                if(DEBUG) {
                    printf("name: %s\n", prop.name);
                    printf("Global Memory: %lu\n", prop.totalGlobalMem);
                    printf("Shared Memory: %lu\n", prop.sharedMemPerBlock);
                    printf("Registers: %d\n", prop.regsPerBlock);
                    printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
                    printf("totalConstMem: %lu\n", prop.totalConstMem);
                    putchar('\n');
                }
            }
        }
    }

    if(!flag) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(0);

    return true;
}

char** read_vocab(char* voc_file_path, int voc_size) {
    int i, len;
    char** vocab;
    FILE* fp;

    fp = fopen(voc_file_path, "r");
    
    vocab = (char**)malloc(sizeof(char*)*voc_size);
    
    for(i=0; i<voc_size; i++) {
        vocab[i] = (char*)malloc(sizeof(char)*20);
        fgets(vocab[i], 20, fp);
        len = strlen(vocab[i]);
        vocab[i][len-1] = '\0';
    }
    return vocab;
}

void free_vocab(char** vocab, int voc_size) {
    int i;

    for(i=0; i<voc_size; i++) {
        free(vocab[i]);
    }
    free(vocab);
}

void show_vocab(char** vocab, int voc_size, FILE* out) {
    int i;

    for(i=0; i<voc_size; i++) {
        fputs(vocab[i], out);
        fputc('\n', out);
    }
}

#endif

