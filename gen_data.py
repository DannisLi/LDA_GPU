# coding: utf8

import numpy as np


doc_num = 800
top_num = 2
word_num_min = 40
word_num_max = 80
voc_size = 3

# doc-topic distribution
doc_top = np.array([0.6, 0.4])

# topic-word distribution, each row represents a topic
top_word = np.array([[0.2, 0.8, 0.], [0.6, 0., 0.4]])



print (doc_num)
print (voc_size)

for i in range(doc_num):
    word_num = int(np.random.uniform(low=word_num_min, high=word_num_max))
    word_cnts = np.zeros(voc_size, dtype=np.int)

    for j in range(word_num):
        top = np.random.choice(top_num, p=doc_top)
        word = np.random.choice(voc_size, p=top_word[top])
        word_cnts[word] += 1

    for j in range(voc_size):
        if word_cnts[j] > 0:
            print (i+1, j+1, word_cnts[j])


