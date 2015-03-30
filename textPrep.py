import re
import nltk
import scipy
import numpy

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

stopwords = nltk.corpus.stopwords.words('english')

collection = {}

with open('../../Datastore/shakespeare.txt', 'r') as f:
	raw = f.read()
	titles = [t for t in set(re.findall(r'[a-z]+@{1}', raw))]

	histories = ['loverscomplaint@', 'glossary@', 'kinghenryvi@', 'kingrichardiii@', 'venusandadonis@', 'rapeoflucrece@', 'kingrichardii@', 'kingjohn@', 'kinghenryiv@', 'kinghenryv@', 'sonnets@', 'various@', 'kinghenryviii@']
	titles = [title for title in titles if title not in histories]

	lines = raw.splitlines()
	for t in titles:
		key = t[:-1]
		text = [re.sub(r'[a-z]+@{1}[0-9]+[ \t\n]+', '', l) for l in lines if re.match(t+'[0-9]+', l)]
		collection[key] = ' '.join(text)

		with open('../../Datastore/Shakespeare/' + key + '.txt', 'w') as outputfile:
			for l in text:
				outputfile.write('%s\n' % l)

tokens = {}
stemmed = {}
tokenizer = RegexpTokenizer(r'\w+')
st = nltk.stem.PorterStemmer()

with open('glossary.txt', 'r') as f:
	glossary = f.read() 

glossary = tokenizer.tokenize(glossary)
fdist = {}

for key in collection.keys():
	collection[key] = re.sub(r'\n?\n[A-Z]*\t|\n|\t', ' ', collection[key])
	tokens[key] = tokenizer.tokenize(collection[key]) # delete punctuation
	names = [w.lower() for w in set(tokens[key]) if w.isupper()]
	tokens[key] = [w for w in tokens[key] if w.lower() not in names]
	tokens[key] = [w.lower() for w in tokens[key] if w.lower() not in stopwords + glossary] # delete stop words
	#stemmed[key] = [st.stem(w.lower()) for w in tokens[key]]
	


common_words = set(tokens['hamlet'])

for (i, (play, words)) in enumerate(tokens.iteritems()):
	print(i, play, len(common_words))
	if i == 10:
		continue
	else:
		common_words = common_words.intersection(set(words))
	#print(common_words)


for key in tokens.keys():
	tokens[key] = [w for w in tokens[key] if w in common_words]
	fdist[key] = nltk.FreqDist(tokens[key])

# [item[1] for item in fdist[fdist.keys()[0]].items()] # absolute freqeuncy

a = scipy.array([fdist[fdist.keys()[0]].freq(w) for w in common_words]) # relative frequency

for t in fdist.keys()[1:]:
	a = scipy.append(a, [fdist[t].freq(w) for w in common_words])

b = a.reshape((len(common_words), len(fdist.keys())))

kmean_input = b.T
print kmean_input.shape

numpy.savetxt('input.csv', kmean_input, delimiter=',')

rclusters = [2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2]

for i in range(len(rclusters)):
    print (rclusters[i], fdist.keys()[i])
