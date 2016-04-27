import re

collection = {}

f = open('shakespeare.txt', 'r')
raw = f.read()
titles = [title for title in set(re.findall('\w+@', raw))]

lines = raw.splitlines()
for title in titles:
    play = title[:-1]
    collection[play] = [re.sub(title + '[0-9]+[\t]+', '', l)
                        for l in lines if re.match(title + '[0-9]+', l)]
    with open('plays/' + play + '.txt', 'w') as outputfile:
        for l in collection[play]:
            outputfile.write('{}\n'.format(l))
