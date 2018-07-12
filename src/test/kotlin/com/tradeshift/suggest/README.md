## Comparing results with scikit-learn NB classifier:

With some runnable python code:
```
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
    slist = [s for s in slist if s not in stoppingWords]
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

# gives: Baseline NB classifier test score: 0.6439193
```

Thus we know our model is performing equally good, as we have **EXACTLY THE SAME** score above **0.6439193**, if we 
use the same `[^a-zA-Z]+` regular expressions in our model.

---
## When using \p type regular expression

As the python default re could not handle `\p` style regular expressions, we had to install a `regex` module
in python to try with this regular expression setup.

```
def clean_text_the_same(s):
    s = regex.sub("[^\p{L}\p{N}]+", ' ', s) 
    s = s.strip()
    s = s.lower()
    
    slist = s.split(' ')
    slist = [s for s in slist if s not in stoppingWords]
    s = ' '.join(slist)
    
    return s

t_test, y_test = read_data("20newsgroup_test.txt")
t_train, y_train = read_data("20newsgroup_train.txt")

t_train = list(map(clean_text_the_same, t_train))
t_test = list(map(clean_text_the_same, t_test))

countVectorizer = CountVectorizer(min_df=1)
x_train = countVectorizer.fit_transform(t_train)
x_test = countVectorizer.transform(t_test)

nb = MultinomialNB(alpha=1.0)
nb.fit(x_train, y_train)
print('Baseline NB classifier test score: {0:0.7f}'.format(nb.score(x_test, y_test)))

# gives: Baseline NB classifier test score: 0.6427244
```

however out model gives: **0.641795**

I assume the difference is due to the different implementation of regex functions.

---

## Other scores

Scikit-learn's NB scores can be increased to **0.6758** with some text cleaning and using scikit-learn's tf-idf vectorizer.

```
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

print(len(newsgroups_train.target))
print(len(newsgroups_test.target))

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

# Gives: 0.8278889894475222
```

With 20newsgroup data directly from scikit-learn package, the data has extra stuff such as 
headers. Using that the scikit-learn NB gives score **0.788** with simple count and **0.828** with tf-idf.

Tried this scikit-learn data on our model as well and we have score above **0.79**.

---

Some helper code in case we want to export data from scikit-learn again.
```
with open('../resources/20newsgroup_train.txt', 'w') as f:
    for label, text in zip(newsgroups_train.target, newsgroups_train.data):
        f.write(str(label) + ' ' + text.replace('\n', ' ') + '\n')
        
with open('../resources/20newsgroup_test.txt', 'w') as f:
    for label, text in zip(newsgroups_test.target, newsgroups_test.data):
        f.write(str(label) + ' ' + text.replace('\n', ' ') + '\n')
```
