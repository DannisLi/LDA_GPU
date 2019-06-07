
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#include "tools.h"
#include "matrix.h"
#include "corpus.h"

#ifndef LDA_H_
#define LDA_H_

#define n_epoch 200

// 功能：随机采样一次多项分布
__host__ int sample_from_multinomial(float* prob, int n) {
    int i;
    float sum = 0., acm, u;

    for(i=0; i<n; i++) {
        sum += prob[i];
    }
    
    // 生成U[0,1]的一个样本
    u = (float)rand() / RAND_MAX * sum;

    acm = 0.;
    for(i=0; i<n-1; i++) {
        acm += prob[i];
        if(acm >= u) {
            return i;
        }
    }

    return n-1;
}


__device__ int sample_from_multinomial_on_device(float* prob, int n, curandState* state_ptr) {
    int i;
    float sum = 0., acm, u;

    for(i=0; i<n; i++) {
        sum += prob[i];
    }
    
    // 生成U[0,1]的一个样本
    u = curand_uniform(state_ptr) * sum;

    acm = 0.;
    for(i=0; i<n-1; i++) {
        acm += prob[i];
        if(acm >= u) {
            return i;
        }
    }

    return n-1;
}


double LDA_evaluate(CORPUS* corpus, MATRIX* topic_doc_cnts, MATRIX* topic_word_cnts) {
    int i, j, k, w, topic_num = topic_doc_cnts->row, doc_num = corpus->doc_num;
    double perp, p_word, log_p_doc, p_doc_topic, p_topic_word;    
    MATRIX* topic_doc_probs;
    MATRIX* topic_word_probs;

    topic_doc_probs = MATRIX_copy(topic_doc_cnts);
    topic_word_probs = MATRIX_copy(topic_word_cnts);
    
    // 归一化概率
    for(i=0; i<doc_num; i++) {
        MATRIX_col_normalize(topic_doc_probs, i);
    }
    for(i=0; i<topic_num; i++) {
        MATRIX_row_normalize(topic_word_probs, i);
    }

    perp = 0.;

    for(i=0; i<corpus->doc_num; i++) {
        log_p_doc = 0;
        for(j=corpus->doc_index[i]; j<corpus->doc_index[i+1]; j++) {
            w = corpus->words[j];
            p_word = 0;
            for(k=0; k<topic_num; k++) {
                p_doc_topic = MATRIX_read(topic_doc_probs, k, i);
                p_topic_word = MATRIX_read(topic_word_probs, k, w);
                p_word += (double)p_doc_topic * (double)p_topic_word;
            }
            log_p_doc += corpus->cnts[j] * log(p_word);
        }
        perp += log_p_doc;
    }

    return perp;
}



// 功能：串行化LDA算法
void serial_LDA(CORPUS* corpus, int topic_num, float alpha, float beta, MATRIX* topic_doc_cnts,  MATRIX* topic_word_cnts) {
    int i, j, k, h, m, w, z, cnt, epoch, doc_num = corpus->doc_num;
    float prob[topic_num];
    int temp[topic_num];
    int topic_cnts[topic_num];
    float voc_size_times_beta = corpus->voc_size * beta;
    float **topic_doc_cnts_data = topic_doc_cnts->data, **topic_word_cnts_data = topic_word_cnts->data;
    int* doc_word_topics[topic_num];    // 每篇文档中每个单词的主题分布

    // 初始化主题计数
    memset(topic_cnts, 0 ,sizeof(int)*topic_num);
    for(i=0; i<topic_num; i++) {
        doc_word_topics[i] = (int*)malloc(sizeof(int)*corpus->doc_index[doc_num]);
        memset(doc_word_topics[i], 0, sizeof(int)*corpus->doc_index[doc_num]);
    }

    // 随机分配主题
	for(i=0; i<doc_num; i++) {
        for(j=corpus->doc_index[i]; j<corpus->doc_index[i+1]; j++) {
            w = corpus->words[j];
            cnt = corpus->cnts[j];
            for(k=0; k<cnt; k++) {
                z = rand() % topic_num;
                topic_cnts[z] += 1;
                doc_word_topics[z][j] += 1;
                topic_doc_cnts_data[z][i] += 1;
                topic_word_cnts_data[z][w] += 1;
            }
        }
    }

    printf("before perp: %lf\n", LDA_evaluate(corpus, topic_doc_cnts, topic_word_cnts));
    
    // 串行化的LDA算法
    for(epoch=1; epoch<=n_epoch; epoch++) {
        for(i=0; i<doc_num; i++) {
            for(j=corpus->doc_index[i]; j<corpus->doc_index[i+1]; j++) {
                w = corpus->words[j];
                for(k=0; k<topic_num; k++) {
                    temp[k] = doc_word_topics[k][j];
                }
                for(k=0; k<topic_num; k++) {
                    for(h=0; h<temp[k]; h++) {
                        // 计算每个主题的条件概率
                        for(m=0; m<topic_num; m++) {
                            if(m!=k) {
                                prob[m] = (topic_doc_cnts_data[m][i] + alpha) * (topic_word_cnts_data[m][w] + beta) / (topic_cnts[m] + voc_size_times_beta);
                            } else {
                                prob[m] = (topic_doc_cnts_data[m][i] - 1 + alpha) * (topic_word_cnts_data[m][w] - 1 + beta) / (topic_cnts[m] - 1 + voc_size_times_beta);
                            }
                        }
                        // 采样新的主题
                        z = sample_from_multinomial(prob, topic_num);
                        if(k!=z) {
                            // 修改相关计数
                            // 删除原来的主题标记
                            topic_doc_cnts_data[k][i] -= 1;
                            topic_word_cnts_data[k][w] -= 1;
                            topic_cnts[k]--;
                            doc_word_topics[k][j]--;
                            // 添加新的主题标记
                            topic_doc_cnts_data[z][i] += 1;
                            topic_word_cnts_data[z][w] += 1;
                            topic_cnts[z]++;
                            doc_word_topics[z][j]++;
                        }
                    }
                }
            }
        }
    }
    printf("after perp: %lf\n", LDA_evaluate(corpus, topic_doc_cnts, topic_word_cnts));


    for(i=0; i<topic_num; i++) {
        free(doc_word_topics[i]);
    }
}


__constant__ int corpus_doc_index[4000];

__global__ static void parallel_LDA_kernel(CORPUS* corpus, int topic_num, float alpha, float beta, float** topic_doc_cnts, float*** topic_word_cnts_p, int* topic_cnts, int** doc_word_topics_d, int seed) {
    // 矩阵topic_doc_cnts的行指针
    __shared__ float* topic_doc_cnts_rows[8];
    // 每个矩阵topic_word_cnts的行指针
    __shared__ float* topic_word_cnts_rows_p[thread_num+1][8];
    // 主题计数向量topic_cnts
    __shared__ int topic_cnts_p[thread_num][8];
    // 名字代换
    int doc_num = corpus->doc_num, voc_size = corpus->voc_size, *corpus_words = corpus->words;
    // 条件概率
    float prob[8];
    // 提前计算voc_size * beta
    float voc_size_times_beta = corpus->voc_size * beta;
    // 暂存doc_word_topics
    int temp[8];
    // 循环控制变量
    int i, j, k, h, m, epoch;
    // token的主题
    int z;
    // token的值
    int w;
    // cuda产生随机数的输入参数
    curandState state;
    
    // 初始化随机种子
    curand_init(seed, threadIdx.x, 0, &state);

    // shared memory: topic_doc_cnts_rows
    for(i=threadIdx.x; i<topic_num; i+=thread_num) {
        topic_doc_cnts_rows[i] = topic_doc_cnts[i];
    }
    // shared memory: topic_word_cnts_rows_p
    for(i=0; i<topic_num; i++) {
        topic_word_cnts_rows_p[threadIdx.x][i] = topic_word_cnts_p[threadIdx.x][i];
    }
    if(threadIdx.x == 0) {
        for(i=0; i<topic_num; i++) {
            topic_word_cnts_rows_p[thread_num][i] = topic_word_cnts_p[thread_num][i];
        }
    }
    // shared memory: topic_cnts_p
    for(i=0; i<topic_num; i++) {
        topic_cnts_p[threadIdx.x][i] = topic_cnts[i];
    }
    __syncthreads();
    
    // 串行化的LDA算法
    for(epoch=1; epoch<=n_epoch; epoch++) {
        // 每个线程负责一部分文档
        for(i=threadIdx.x; i<doc_num; i+=thread_num) {
            for(j=corpus_doc_index[i]; j<corpus_doc_index[i+1]; j++) {
                w = corpus_words[j];
                for(k=0; k<topic_num; k++) {
                    temp[k] = doc_word_topics_d[k][j];
                }
                for(k=0; k<topic_num; k++) {
                    for(h=0; h<temp[k]; h++) {
                        // 计算每个主题的条件概率
                        for(m=0; m<topic_num; m++) {
                            if(m!=k) {
                                prob[m] = (topic_doc_cnts_rows[m][i] + alpha) * (topic_word_cnts_rows_p[threadIdx.x][m][w] + beta) / (topic_cnts_p[threadIdx.x][m] + voc_size_times_beta);
                            } else {
                                prob[m] = (topic_doc_cnts_rows[m][i] - 1 + alpha) * (topic_word_cnts_rows_p[threadIdx.x][m][w] - 1 + beta) / (topic_cnts_p[threadIdx.x][m] - 1 + voc_size_times_beta);
                            }
                        }
                        // 采样新的主题
                        z = sample_from_multinomial_on_device(prob, topic_num, &state);
                        if(z!=k) {
                            // 删除原来的主题标记
                            topic_doc_cnts_rows[k][i] -= 1;
                            topic_word_cnts_rows_p[threadIdx.x][k][w] -= 1;
                            topic_cnts_p[threadIdx.x][k] -= 1;
                            doc_word_topics_d[k][j] -= 1;
                            // 添加新的主题标记
                            topic_doc_cnts_rows[z][i] += 1;
                            topic_word_cnts_rows_p[threadIdx.x][z][w] += 1;
                            topic_cnts_p[threadIdx.x][z] += 1;
                            doc_word_topics_d[z][j] += 1;
                        }
                    }
                }
            }
        }
        if(epoch%10==0) {
            // 统一一次计数矩阵
            // MATRIX_sub(topic_word_cnts_p[threadIdx.x], topic_word_cnts_p[thread_num]);
            for(i=0; i<topic_num; i++) {
                for(j=0; j<voc_size; j++) {
                    topic_word_cnts_rows_p[threadIdx.x][i][j] -= topic_word_cnts_rows_p[thread_num][i][j];
                }
            }
            __syncthreads();
            // topic_word_cnts_p[0] = \sum_{i=0}^{thread_num-1} topic_word_cnts_p[i]
            for(i=thread_num/2; i>0; i/=2) {
                if(threadIdx.x < i) {
                    for(j=0; j<topic_num; j++) {
                        for(k=0; k<voc_size; k++) {
                            topic_word_cnts_rows_p[threadIdx.x][j][k] += topic_word_cnts_rows_p[threadIdx.x+i][j][k];
                        }
                    }
                }
                __syncthreads();
            }
            // topic_word_cnts_p[thread_num] = topic_word_cnts_p[0] + topic_word_cnts_p[thread_num]
            for(i=0; i<topic_num; i++) {
                for(j=threadIdx.x; j<voc_size; j+=thread_num) {
                    topic_word_cnts_rows_p[thread_num][i][j] += topic_word_cnts_rows_p[0][i][j];
                }
            }
            __syncthreads();
            // topic_word_cnts_p[i] = topic_word_cnts_p[thread_num], forall i=0 ,..., thread_num-1
            for(i=0; i<topic_num; i++) {
                for(j=0; j<voc_size; j++) {
                    topic_word_cnts_rows_p[threadIdx.x][i][j] = topic_word_cnts_rows_p[thread_num][i][j];
                }
            }
            // 重新计算每个线程的topic_cnts
            for(j=0; j<topic_num; j++) {
                topic_cnts_p[threadIdx.x][j] = 0;
                for(k=0; k<doc_num; k++) {
                    topic_cnts_p[threadIdx.x][j] += topic_doc_cnts_rows[j][k];
                }
            }
            __syncthreads();
        }
    }
}


void parallel_LDA(CORPUS* corpus_h, int topic_num, float alpha, float beta, MATRIX* topic_doc_cnts_h, MATRIX* topic_word_cnts_h) { 
    // topic doc
    float **topic_doc_cnts_d;
    // topic word
    float ***topic_word_cnts_d_p, **topic_word_cnts_tmp[thread_num + 1];    // 每个线程有单独的计数
    // corpus
    CORPUS* corpus_d;
    // topic counts
    int topic_cnts_h[topic_num], *topic_cnts_d;
    // doc-word-topic
    int *doc_word_topics_h[topic_num], **doc_word_topics_d, *doc_word_topics_tmp[topic_num];
    int i, j, k, w, z, cnt, doc_num = corpus_h->doc_num;

    // 初始化topic counts
    memset(topic_cnts_h, 0 ,sizeof(int)*topic_num);

    // 初始化doc-word-topic
    for(i=0; i<topic_num; i++) {
        doc_word_topics_h[i] = (int*)malloc(sizeof(int)*corpus_h->doc_index[doc_num]);
        memset(doc_word_topics_h[i], 0, sizeof(int)*corpus_h->doc_index[doc_num]);
    }
    cudaMalloc(&doc_word_topics_d, sizeof(int*)*topic_num);
    
    // 随机分配主题
    for(i=0; i<doc_num; i++) {
        for(j=corpus_h->doc_index[i]; j<corpus_h->doc_index[i+1]; j++) {
            w = corpus_h->words[j];
            cnt = corpus_h->cnts[j];
            for(k=0; k<cnt; k++) {
                z = rand() % topic_num;
                topic_cnts_h[z]++;
                doc_word_topics_h[z][j] += 1;
                topic_doc_cnts_h->data[z][i] += 1;
                topic_word_cnts_h->data[z][w] += 1;
            }
        }
    }
    
    // 在设备端创建矩阵topic_doc_cnts，该矩阵所有线程共享
    topic_doc_cnts_d = MATRIX_create_core_on_device(topic_doc_cnts_h);
    
    // 为每个线程，在设备端创建矩阵topic_word_cnts，每个线程独有一个矩阵，并且有一个矩阵保存原始值
    for(i=0; i<=thread_num; i++) {
        topic_word_cnts_tmp[i] = MATRIX_create_core_on_device(topic_word_cnts_h);
    }
    cudaMalloc(&topic_word_cnts_d_p, sizeof(float**)*(thread_num+1));
    cudaMemcpy(topic_word_cnts_d_p, topic_word_cnts_tmp, sizeof(float**)*(thread_num+1), cudaMemcpyHostToDevice);

    // 在设备端创建主题计数topic_cnts
    cudaMalloc(&topic_cnts_d, sizeof(int)*topic_num);
    cudaMemcpy(topic_cnts_d, topic_cnts_h, sizeof(int)*topic_num, cudaMemcpyHostToDevice);

    // 在设备端创建每个token的主题记录doc_word_topics
    for(i=0; i<topic_num; i++) {
        cudaMalloc(&(doc_word_topics_tmp[i]), sizeof(int)*corpus_h->doc_index[doc_num]);
        cudaMemcpy(doc_word_topics_tmp[i], doc_word_topics_h[i], sizeof(int)*corpus_h->doc_index[doc_num], cudaMemcpyHostToDevice);
    }
    cudaMalloc(&doc_word_topics_d, sizeof(int*)*topic_num);
    cudaMemcpy(doc_word_topics_d, doc_word_topics_tmp, sizeof(int*)*topic_num, cudaMemcpyHostToDevice);

    // 将语料库移动到设备端
    corpus_d = CORPUS_create_the_same_on_device(corpus_h);

    // 将corpus的corpus_doc_index部分放到costant memory
    cudaMemcpyToSymbol(corpus_doc_index, corpus_h->doc_index, sizeof(int)*(corpus_h->doc_num+1));
    
    // printf("before kernel\n");
    printf("before perp: %lf\n", LDA_evaluate(corpus_h, topic_doc_cnts_h, topic_word_cnts_h));
    // 在设备端运行LDA算法
    parallel_LDA_kernel<<<1, thread_num>>>(corpus_d, topic_num, alpha, beta, topic_doc_cnts_d, topic_word_cnts_d_p, topic_cnts_d, doc_word_topics_d, rand());
    // printf("after kernel\n");

    // 拷贝设备端的计数矩阵
    MATRIX_move_core_to_host(topic_doc_cnts_h, topic_doc_cnts_d);
    MATRIX_move_core_to_host(topic_word_cnts_h, topic_word_cnts_tmp[thread_num]);
    // MATRIX_move_core_to_host(topic_word_cnts_h, topic_word_cnts_tmp[0]);

    // 释放设备端矩阵topic_doc_cnts
    MATRIX_free_core_device(topic_doc_cnts_d, topic_num);

    // 释放设备上的topic_word_cnts
    cudaFree(topic_word_cnts_d_p);
    for(i=0; i<=thread_num; i++) {
        MATRIX_free_core_device(topic_word_cnts_tmp[i], topic_num);
    }

    // 释放设备上的topic_cnts
    cudaFree(topic_cnts_d);
    
    // 释放设备上的doc_word_topics_d
    cudaFree(doc_word_topics_d);
    for(i=0; i<topic_num; i++) {
        cudaFree(doc_word_topics_tmp[i]);
    }
    
    // 释放设备上的corpus
    CORPUS_free_device(corpus_d);
    
    printf("after perp: %lf\n", LDA_evaluate(corpus_h, topic_doc_cnts_h, topic_word_cnts_h));

}

#endif 
