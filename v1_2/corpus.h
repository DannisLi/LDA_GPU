
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CORPUS_H_
#define CORPUS_H_


// 采用稀疏矩阵存储格式CRS，存储语料库
typedef struct CORPUS {
    int doc_num, voc_size;
    int* doc_index;
    int* docs;
    int* words;
    int* cnts;
} CORPUS;

// 输入：语料库对象指针
// 功能：释放主机端的语料库
void CORPUS_free(CORPUS* corpus) {
    free(corpus->doc_index);
    free(corpus->docs);
    free(corpus->words);
    free(corpus->cnts);
    free(corpus);
}

// 功能：从文档中读取语料库并返回；若失败，返回NULL
CORPUS* CORPUS_from_file(char* file_path) {
    FILE* fp;
    int i, row_index[10000], rows[1000000], cols[1000000], vals[1000000];    // 使用三元组暂存
    int doc, word, cnt, last_doc, doc_num, voc_size;
    CORPUS* corpus;
    
    // 打开文件
    if((fp=fopen(file_path, "r")) == NULL) {
        return NULL;
    }

    // 读取头部信息
    fscanf(fp, "%d%d", &doc_num, &voc_size);
    
    // 逐个单词插入
    i = 0;
    last_doc = 0;
    row_index[0] = 0;
    while(!feof(fp)) {
        fscanf(fp, "%d%d%d\n", &doc, &word, &cnt);
        // printf ("%d %d %d\n", doc, word, cnt);
        doc--;
        word--;

        if(last_doc!=doc) {
            row_index[doc] = i;
            last_doc = doc;
        }
        rows[i] = doc;
        cols[i] = word;
        vals[i] = cnt;
        i++;
    }
    row_index[doc_num] = i;
    fclose(fp);

    corpus = (CORPUS*)malloc(sizeof(CORPUS));

    corpus->doc_num = doc_num;
    corpus->voc_size = voc_size;
    
    corpus->doc_index = (int*)malloc(sizeof(int)*(doc_num+1));
    memcpy(corpus->doc_index, row_index, sizeof(int)*(doc_num+1));

    corpus->docs = (int*)malloc(sizeof(int)*i);
    memcpy(corpus->docs, rows, sizeof(int)*i);

    corpus->words = (int*)malloc(sizeof(int)*i);
    memcpy(corpus->words, cols, sizeof(int)*i);

    corpus->cnts = (int*)malloc(sizeof(int)*i);
    memcpy(corpus->cnts, vals, sizeof(int)*i);
    
    return corpus;
}

// 功能：打印语料库
void CORPUS_show(CORPUS* corpus, FILE* out) {
    int i, j;

    // 打印文档数量、单词表长度
    fprintf(out, "%d\n%d\n", corpus->doc_num, corpus->voc_size);
    
    // 打印每篇文档
    for(i=0; i<corpus->doc_num; i++) {
        for(j=corpus->doc_index[i]; j<corpus->doc_index[i+1]; j++) {
            fprintf(out, "%d %d %d\n", i+1, corpus->words[j]+1, corpus->cnts[j]);
        }
    }
}

// 功能：移动语料库到设备端
CORPUS* CORPUS_create_the_same_on_device(CORPUS* corpus_h) {
    CORPUS corpus_tmp, *corpus_d;
    int len = corpus_h->doc_index[corpus_h->doc_num];

    corpus_tmp.doc_num = corpus_h->doc_num;
    corpus_tmp.voc_size = corpus_h->voc_size;

    cudaMalloc(&(corpus_tmp.doc_index), sizeof(int)*(corpus_h->doc_num+1));
    cudaMemcpy(corpus_tmp.doc_index, corpus_h->doc_index, sizeof(int)*(corpus_h->doc_num+1), cudaMemcpyHostToDevice);
    
    cudaMalloc(&(corpus_tmp.docs), sizeof(int)*len);
    cudaMemcpy(corpus_tmp.docs, corpus_h->docs, sizeof(int)*len, cudaMemcpyHostToDevice);
    
    cudaMalloc(&(corpus_tmp.words), sizeof(int)*len);
    cudaMemcpy(corpus_tmp.words, corpus_h->words, sizeof(int)*len, cudaMemcpyHostToDevice);
    
    cudaMalloc(&(corpus_tmp.cnts), sizeof(int)*len);
    cudaMemcpy(corpus_tmp.cnts, corpus_h->cnts, sizeof(int)*len, cudaMemcpyHostToDevice);

    cudaMalloc(&corpus_d, sizeof(CORPUS));
    cudaMemcpy(corpus_d, &corpus_tmp, sizeof(CORPUS), cudaMemcpyHostToDevice);

    return corpus_d;
}


// 功能：释放设备端的语料库
void CORPUS_free_device(CORPUS* corpus_d) {
    CORPUS corpus_tmp;

    cudaMemcpy(&corpus_tmp, corpus_d, sizeof(CORPUS), cudaMemcpyDeviceToHost);
    cudaFree(corpus_tmp.doc_index);
    cudaFree(corpus_tmp.docs);
    cudaFree(corpus_tmp.words);
    cudaFree(corpus_tmp.cnts);
    cudaFree(corpus_d);
}

// 功能：移动语料库到主机端
CORPUS* CORPUS_create_the_same_on_host(CORPUS* corpus_d) {
    CORPUS corpus_tmp, *corpus_h;
    int *doc_index, *cols, *vals;
    
    cudaMemcpy(&corpus_tmp, corpus_d, sizeof(CORPUS), cudaMemcpyDeviceToHost);
    
    doc_index = (int*)malloc(sizeof(int)*(corpus_tmp.doc_num+1));
    cudaMemcpy(doc_index, corpus_tmp.doc_index, sizeof(int)*(corpus_tmp.doc_num+1), cudaMemcpyDeviceToHost);

    cols = (int*)malloc(sizeof(int)*doc_index[corpus_tmp.doc_num]);
    vals = (int*)malloc(sizeof(int)*doc_index[corpus_tmp.doc_num]);
    cudaMemcpy(cols, corpus_tmp.words, sizeof(int)*doc_index[corpus_tmp.doc_num], cudaMemcpyDeviceToHost);
    cudaMemcpy(vals, corpus_tmp.cnts, sizeof(int)*doc_index[corpus_tmp.doc_num], cudaMemcpyDeviceToHost);

    corpus_h = (CORPUS*)malloc(sizeof(CORPUS));
    corpus_h->doc_num = corpus_tmp.doc_num;
    corpus_h->voc_size = corpus_tmp.voc_size;
    corpus_h->doc_index = doc_index;
    corpus_h->words = cols;
    corpus_h->cnts = vals;

    return corpus_h;
}

#endif