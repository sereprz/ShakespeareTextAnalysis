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
import scipy

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
        raw_freq[title][word] = documents[title].count(word)

idf_weights = {}
for word in idf_counts.keys():
    idf_weights[word] = math.log(len(documents)/idf_counts[word])

weights = {}
for title in raw_freq.keys():
    weights[title] = {}
    for word in raw_freq[title]:
        weights[title][word] = raw_freq[title][word]*idf_weights[word]

unique_words = np.array(sorted(idf_counts.keys()))
titles = np.array(sorted(documents.keys()))

weights_matrix = []
for title in titles:
    this_play_weights = weights[title]
    row = [this_play_weights[word] for word in unique_words]
    weights_matrix.append(row)

weights_matrix = np.array(weights_matrix)

print "..done in %f s" % (time() - t0)

##################
##### KMeans #####
##################

def allocate_to_centroids(dat, centroids):
    clusters = []
    for datapoint in dat:
        dist = np.array([scipy.spatial.distance.cosine(datapoint, centroid) for centroid in centroids])
        clusters.append(dist.argmin())
    return(np.array(clusters))

def recalculate_centroids(dat, clusters):
    new_centroids = []
    for i in set(clusters):
        new_centroids.append(dat[np.array([index for index, group in enumerate(clusters) if group == i]),].mean(axis = 0))
    return(np.array(new_centroids))

def km(dat, k):
    """ 
    returns a partition of dat in k groups
    uses kmeans algorithm with cosine similarity
    dat = numpy.matrix object (n x m, n data points, m variables)
    k = int, number of centers
    """

    # initialize with k random data points
    centroids = random.sample(dat, k)
    # allocate to centroid of minimum cosine distance
    clusters = allocate_to_centroids(dat, centroids)
    # recalculate centroids as average over the cluster
    new_centroids = recalculate_centroids(dat, clusters)

    iternum = 0

    while not np.array_equal(centroids, new_centroids) and iternum <= 20:
        iternum += 1
        centroids = new_centroids
        clusters = allocate_to_centroids(dat, centroids)
        new_centroids = recalculate_centroids(dat, clusters)
        print 'iteration', iternum

    return(clusters)

test = km(weights_matrix, 2)


## find indeces:
# groups = [0, 0, 0, 1, 0, 1, 1, 0]  note, this a list
# titles[np.array([index for index, group in enumerate(groups) if group == 1])]
# also, for matrices
# weights_matrix[np.array([index for index, group in enumerate(groups) if group == 1]), :10]
# if groups = np.array([0, 0, 0, 1, 0, 1, 1, 0], int)
# weights_matrix[[index for index, group in enumerate(groups) if group == 1], :10]