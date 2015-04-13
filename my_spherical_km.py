############################
##### Spherical KMeans #####
############################
import random
import numpy as np

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
    clusters = range(k)
    while len(clusters) < dat.shape[0]:
        clusters.extend(range(k))
    clusters = np.array(clusters[:dat.shape[0]])
    random.shuffle(clusters)

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
