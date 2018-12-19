import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def read_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    splitedLines = map(lambda l: l.split(' ', 1), lines)
    t = list(map(lambda s: s[1], splitedLines))

    splitedLines = map(lambda l: l.split(' ', 1), lines)
    y = list(map(lambda s: s[0], splitedLines))

    print(len(t))
    return t, y

def clean_text(s):
    s = re.sub("[^a-zA-Z]+", ' ', s)
    s = s.strip()
    s = s.lower()

    slist = s.split(' ')
    # slist = [s for s in slist if s not in stoppingWords]
    s = ' '.join(slist)

    return s

t_test, y_test = read_data("20newsgroup_test.txt")
t_train, y_train = read_data("20newsgroup_train.txt")

t_train = list(map(clean_text, t_train))
t_test = list(map(clean_text, t_test))

countVectorizer = CountVectorizer(min_df=1)
x_train = countVectorizer.fit_transform(t_train)
x_test = countVectorizer.transform(t_test)

nb = MultinomialNB(alpha=1.0)
nb.fit(x_train, y_train)
print('Baseline NB classifier test score: {0:0.7f}'.format(nb.score(x_test, y_test)))
