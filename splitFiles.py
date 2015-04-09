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

	histories = ['glossary@', 'various@', 'sonnets@'] #, 'loverscomplaint@', 'kinghenryvi@', 'kingrichardiii@', 'venusandadonis@', 'rapeoflucrece@', 'kingrichardii@', 'kingjohn@', 'kinghenryiv@', 'kinghenryv@','kinghenryviii@']
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


	with open('../../Datastore/Shakespeare/Normalized/' + key + '_normalized.txt', 'w') as outputfile:
		for w in tokens[key]: #for w in stemmed[key]:
			outputfile.write('%s ' % w)






