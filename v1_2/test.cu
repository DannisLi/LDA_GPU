#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#include "corpus.h"

#define n_epoch 400




int main()
{
    CORPUS* corpus = CORPUS_from_file("../data/nips.corpus");
    printf("aaa\n");
    CORPUS_show(corpus, stdout);
    CORPUS_free(corpus);
    
    return 0;
}


