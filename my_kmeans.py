import random as random
import scipy
import numpy as np

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

    while not np.array_equal(centroids, new_centroids):
        iternum += 1
        centroids = new_centroids
        clusters = allocate_to_centroids(dat, centroids)
        new_centroids = recalculate_centroids(dat, clusters)
        print 'iteration', iternum

    return(clusters)