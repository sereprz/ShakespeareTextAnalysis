from __future__ import division

import glob
import nltk
import math
import re
import pandas as pd
import random
import numpy as np


path = '../../Datastore/Shakespeare/Normalized/*.txt'
documents = {}

for filename in glob.glob(path):
	with open(filename, 'r') as f:
		text = f.read()
		key = re.sub( r'/', '', str(re.findall(r'/[a-z]+', filename)[0]))
		documents[key] = nltk.word_tokenize(text)


def ts(w, d):
	return(d.count(w))

def idf(w, D):
	# inverse document frequency
	# 
	return( math.log(len(D)/sum([1 for key in D.keys() if w in D[key]])))

def tfidf(w, d, D):
	# term freqency-inverse ddocument frequency
	# http://en.wikipedia.org/wiki/Tf%E2%80%93idf
	return(ts(w,d)*idf(w,D))

def cosine_similarity(d1, d2):
	# cosine similarity between two vectors
	return(np.dot(d1, d2)/(np.linalg.norm(d1)*np.linalg.norm(d2)))

wlist = []

for key in documents.keys():
	wlist = wlist + documents[key]

wlist = list(set(wlist)) # all the words in the corpus

# m = max([idf(w, documents) for w in wlist])
# max_tsdf = [w for w in wlist if idf(w, documents) == m]
# words with max idf only appear in one of the plays, which makes the weights useless in terms of
# distance and clustering

words_sample = random.sample(wlist, int(len(wlist)*0.02))

weights = {}
for key in documents.keys():
	weights[key] = [tfidf(w, documents[key], documents) for w in words_sample]

## just testing
for key in weights.keys():
	print key, 1/cosine_similarity(weights['tamingoftheshrew'], weights[key])

df = pd.DataFrame(weights).T

