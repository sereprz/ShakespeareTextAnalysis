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
import lda

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

freq_matrix = []
for title in titles:
    this_play_freq = raw_freq[title]
    row = [this_play_freq[word] for word in unique_words]
    freq_matrix.append(row)

weights_matrix = np.array(weights_matrix)  # k-means input

freq_matrix = np.array(freq_matrix) # LDA input

print "..done in %f s" % (time() - t0)

##################
##### KMeans #####
##################

# The algorithm has converged when the assignments no longer change. Since both steps optimize the WCSS objective, and there only exists a finite number of such partitionings, the algorithm must converge to a (local) optimum. There is no guarantee that the global optimum is found using this algorithm.

# The algorithm is often presented as assigning objects to the nearest cluster by distance. The standard algorithm aims at minimizing the WCSS objective, and thus assigns by "least sum of squares", which is exactly equivalent to assigning by the smallest Euclidean distance. Using a different distance function other than (squared) Euclidean distance may stop the algorithm from converging.[citation needed] Various modifications of k-means such as spherical k-means and k-medoids have been proposed to allow using other distance measures.

def allocate_to_centroids(dat, centroids):
    clusters = []
    for datapoint in dat:
        dist = np.array([scipy.spatial.distance.euclidean(datapoint, centroid) for centroid in centroids])
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

    while not np.array_equal(centroids, new_centroids):
        iternum += 1
        centroids = new_centroids
        clusters = allocate_to_centroids(dat, centroids)
        new_centroids = recalculate_centroids(dat, clusters)
        print 'iteration', iternum

    return(clusters)
    
model1 = km(weights_matrix, 2)

############################
##### Spherical KMeans #####
############################

def recalculate_concept_vectors(dat, clusters):
    concept_vectors = []
    for cluster in set(clusters):
        cluster_elements = dat[np.array([index for index, group in enumerate(clusters) if group == cluster]),]
        sum_of_elements = cluster_elements.sum(axis = 0)
        concept_vectors.append(sum_of_elements/np.linalg.norm(sum_of_elements))
    return(np.array(concept_vectors))

def allocate_to_concept_vectors(dat, concept_vectors):
    clusters = []
    for datapoint in dat:
        dist = np.array([np.dot(datapoint, concept) for concept in concept_vectors])
        clusters.append(dist.argmax())
    return(clusters)

def objective_function(dat, clusters):
    cluster_qualities = []
    for cluster in set(clusters):
        cluster_elements = dat[np.array([index for index, group in enumerate(clusters) if group == cluster]),]
        cluster_qualities.append(np.linalg.norm(cluster_elements.sum(axis = 0)))
    return(sum(cluster_qualities))

def SPKM(dat, k):
    """ spherical k-means
    dat = numpy.matrix object (n x m, n data points, m variables)
    k = int, number of centers
    """

    # set first partition using km
    clusters = np.zeros(dat.shape[0])
    clusters[random.sample(range(dat.shape[0]), int(dat.shape[0]/2))] = 1
    concept_vectors = recalculate_concept_vectors(dat, clusters)
    t = 0
    new_clusters = allocate_to_concept_vectors(dat, concept_vectors)
    new_concept_vectors = recalculate_concept_vectors(dat, clusters)

    while objective_function(dat, new_clusters) - objective_function(dat, clusters) > 0:
        t += 1
        clusters = new_clusters
        concept_vectors = new_concept_vectors
        clusters = allocate_to_concept_vectors(dat, concept_vectors)
        concept_vectors = recalculate_concept_vectors(dat, clusters)
        print 'iter', t, 'delta =', objective_function(dat, new_clusters) - objective_function(dat, clusters)
    return(np.array(clusters))


model2 = SPKM(weights_matrix, 2)

###############
##### LDA #####
###############

model3 = lda.LDA(n_topics=5, n_iter=500, random_state=1)
model3.fit(freq_matrix)

topic_word = model3.topic_word_

n_top_words = 10

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(unique_words)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model3.doc_topic_

for n in range(10):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}\n{}...".format(n, topic_most_pr, titles[n][:50]))

f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([1, 3, 4, 8, 9]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 5)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Document {}".format(k))

ax[4].set_xlabel("Topic")

plt.tight_layout()
plt.show()