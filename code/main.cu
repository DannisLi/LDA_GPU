
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "tools.h"
#include "corpus.h"
#include "LDA.h"



int main()
{
    // 主题数量
    int topic_num;
    // 是否使用GPU
    int USE_GPU;
    // 语料库的名字和路径
    char name[10], *corpus_path, *vocab_path;
    // 单词表
    // char** vocab;
    // 先验参数
    float alpha, beta;
    // 语料库对象
    CORPUS* corpus;
    // 消耗时间
    clock_t used_time;
    // LDA result
	MATRIX *topic_doc_cnts, *topic_word_cnts;
    

    // 输入是否适用GPU
    printf("USE GPU ? (0:don't use  1:use): ");
    scanf("%d", &USE_GPU);

    // 检查CUDA
    if(USE_GPU && !InitCUDA()) {
        printf("No CUDA device!\n");
        return -1;
    }

    
    // 输入语料库名称
    printf("Please input corpus name: ");
    scanf("%s", name);

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

    if(DEBUG) {
        printf("\nYour corpus path: %s\nYour vocab path: %s\n\n", corpus_path, vocab_path);
    }

    printf("Please input number of topics: ");
    scanf("%d", &topic_num);

    if(DEBUG) {
        printf("\nYour topic number: %d\n\n", topic_num);
    }

    // 随机种子
	srand((unsigned)time(NULL));


    // 读取语料库
	if((corpus = CORPUS_from_file(corpus_path)) == NULL) {
        if(DEBUG)
            printf("Fail to read the corpus from file!\n");
        return -1;
    }

	// 计算先验参数：alpha 和 beta
	alpha = 1. / topic_num;
    beta = 1. / corpus->voc_size;

    // 初始化LDA的结果矩阵
    topic_doc_cnts = MATRIX_new(topic_num, corpus->doc_num);
    topic_word_cnts = MATRIX_new(topic_num, corpus->voc_size);
    
    // used_time = clock();
    used_time = time(NULL);
    if(USE_GPU) {
        // parallel_LDA(corpus, topic_num, alpha, beta, topic_doc_cnts, topic_word_cnts);
    } else {
        serial_LDA(corpus, topic_num, alpha, beta, topic_doc_cnts, topic_word_cnts);
    }
    // used_time = clock() - used_time;
    used_time = time(NULL) - used_time;

    // 报告结果
    printf("used time: %lu s\n", used_time);

    // 释放结果
    MATRIX_free(topic_doc_cnts);
    MATRIX_free(topic_word_cnts);
    
    return 0;
}

