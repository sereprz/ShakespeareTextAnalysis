from __future__ import division
from sklearn.cluster import KMeans
from time import time

import collections as c
import glob
import nltk
import math
import re
import pandas as pd
import random
import numpy as np

print 'reading files..'

path = '../../Datastore/Shakespeare/Normalized/*.txt'
documents = {}

for filename in glob.glob(path):
    with open(filename, 'r') as f:
        text = f.read()
        key = re.sub( r'/', '', str(re.findall(r'/[a-z]+', filename)[0]))
        documents[key] = nltk.word_tokenize(text)

def cosine_similarity(d1, d2):
    # cosine similarity between two vectors
    return(np.dot(d1, d2)/(np.linalg.norm(d1)*np.linalg.norm(d2)))

############################
print 'calculating tf-idf weights'
t0 = time()

idf_counts = c.defaultdict(int)
raw_freq = {}

for title, words in documents.iteritems():
    was_word_seen = c.defaultdict(bool)
    for word in words:
        if not was_word_seen[word]:
            idf_counts[word] += 1
            was_word_seen[word] = True

for title in documents.keys():
    raw_freq[title] = {}
    for word in idf_counts:
        raw_freq[title][word] = documents[key].count(word)

idf_weights = {}
for word in idf_counts.keys():
    idf_weights[word] = math.log(len(documents)/idf_counts[word])

weights = {}
for title in raw_freq.keys():
    weights[title] = {}
    for word in raw_freq[title]:
        weights[title][word] = raw_freq[title][word]*idf_weights[word]

unique_words_sorted = np.array(sorted(idf_counts.keys()))
titles = np.array(sorted(documents.keys()))

weights_matrix = []
for title in titles:
    this_play_weights = weights[title]
    row = [this_play_weights[word] for word in unique_words_sorted]
    weights_matrix.append(row)

weights_matrix = np.matrix(weights_matrix)

print "..done in %f s" % (time() - t0)