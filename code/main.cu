
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "tools.h"
#include "corpus.h"
#include "LDA.h"



// ./a.out corpus_name topic_num USE_GPU n_epoch thread_num
int main(int argc, char** argv) {
    // 主题数量
    int topic_num;
    // 是否使用GPU
    int USE_GPU;
    // 迭代轮数
    int n_epoch;
    // 线程数量
    int thread_num;
    // 语料库的名字和路径
    char name[10], *corpus_path, *vocab_path;
    // 先验参数
    float alpha, beta;
    // 语料库对象
    CORPUS* corpus;
    // 消耗时间
    clock_t used_time;
    // LDA result
    MATRIX *topic_doc_cnts, *topic_word_cnts;
    
    // 处理命令行参数
    strcpy(name, argv[1]);
    topic_num = atoi(argv[2]);
    USE_GPU = atoi(argv[3]);
    n_epoch = atoi(argv[4]);
    if(USE_GPU)
        thread_num = atoi(argv[5]);
    

    // 将名字映射为路径
    if(strcmp(name, "kos") == 0) {
        corpus_path = KOS_COR;
        vocab_path = KOS_VOC;
    } else if(strcmp(name, "nips") == 0) {
        corpus_path = NIPS_COR;
        vocab_path = NIPS_VOC;
    } else if(strcmp(name, "enron") == 0) {
        corpus_path = ENRON_COR;
        vocab_path = ENRON_VOC;
    } else if(strcmp(name, "my") == 0) {
        corpus_path = MY_COR;
        vocab_path = MY_VOC;
    } else {
        printf("Wrong corpus name!\n");
        return -1;
    }


    // 随机种子
	srand((unsigned)time(NULL));

    // 读取语料库
    corpus = CORPUS_from_file(corpus_path);

	// 计算先验参数：alpha 和 beta
	alpha = 1. / topic_num;
    beta = 1. / corpus->voc_size;

    // 初始化LDA的结果矩阵
    topic_doc_cnts = MATRIX_new(topic_num, corpus->doc_num);
    topic_word_cnts = MATRIX_new(topic_num, corpus->voc_size);
    
    used_time = time(NULL);
    if(USE_GPU) {
        cudaSetDevice(0);
        parallel_LDA(corpus, topic_num, alpha, beta, topic_doc_cnts, topic_word_cnts, n_epoch, thread_num);
    } else {
        serial_LDA(corpus, topic_num, alpha, beta, topic_doc_cnts, topic_word_cnts, n_epoch);
    }
    used_time = time(NULL) - used_time;

    // 时间
    printf("%lu\n", used_time);

    // 释放结果
    MATRIX_free(topic_doc_cnts);
    MATRIX_free(topic_word_cnts);
    
    return 0;
}

