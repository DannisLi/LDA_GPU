
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation



with open('../data/my.corpus', 'r') as f:
    doc_num = int(f.readline())
    voc_size = int(f.readline())

    X = np.zeros([doc_num, voc_size], dtype=np.int)
    for line in f.readlines():
        line = line.strip()
        if line != '':
            doc,word,cnt = line.split(' ')
            doc = int(doc) - 1
            word = int(word) - 1
            cnt = int(cnt)
            X[doc,word] += cnt



lda = LatentDirichletAllocation(n_components=2,  n_jobs=-1, max_iter=400, batch_size=500)
lda.fit(X)

print (np.transpose(lda.components_.T / np.sum(lda.components_, axis=1)))

print (lda.perplexity(X))
