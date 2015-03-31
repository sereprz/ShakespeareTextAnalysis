from __future__ import division
from sklearn.cluster import KMeans
from time import time

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


def tf(w, d):
	return(d.count(w))

def idf(w, D):
	# inverse document frequency
	return( math.log(len(D)/sum([1 for key in D.keys() if w in D[key]])))

def tfidf(w, d, D):
	# term freqency-inverse ddocument frequency
	# http://en.wikipedia.org/wiki/Tf%E2%80%93idf
	return(tf(w,d)*idf(w,D))

def cosine_similarity(d1, d2):
	# cosine similarity between two vectors
	return(np.dot(d1, d2)/(np.linalg.norm(d1)*np.linalg.norm(d2)))



############################
print 'calculating tf-idf weights'
t0 = time()

idf_counts = {}
raw_freq = {}

for key in documents.keys():
	seen_in_doc = {}
	for w in documents[key]:
		if not seen_in_doc.get(w, False):
			idf_counts[w] = idf_counts.get(w, 0) + 1
			seen_in_doc[w] = True

for key in documents.keys():
	raw_freq[key] = {}
	for w in idf_counts:
		raw_freq[key][w] = tf(w, documents[key])

idf_weights = {}
for w in idf_counts.keys():
	idf_weights[w] = math.log(len(documents)/idf_counts[w])

weights = {}
for key in raw_freq.keys():
	weights[key] = {}
	for w in raw_freq[key]:
		weights[key][w] = raw_freq[key][w]*idf_weights[w]

print "..done in %f s" % (time() - t0)

## just testing
#for key in weights.keys():
#	print key, 1/cosine_similarity(weights['tamingoftheshrew'], weights[key])

#df = pd.DataFrame(weights).T

#km = KMeans(init='random', n_clusters = 2, n_init = 1, verbose = 0)
#km.fit(df)

#print km.labels_