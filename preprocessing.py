# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 09:19:19 2018

@author: KALIT
"""

import pandas as pd
#read data train
data_asli=pd.read_csv('data_gojek_random_label_3060.csv')

del data_asli['Unnamed: 0']
print len(data_asli)
data_asli.head(30)
data_asli.count()

data_train=data_asli

# convert data to lowercase
import string
for i in range(len(data_train)):
    text = string.lower(data_train['text'].iloc[i])
    data_train['text'].iloc[i]=text
print data_train['text'][0]
data_train.head()

#remove number
import re
pattern=r'[0-9]+'
for i in range(len(data_train)):
    data_train['text'].iloc[i] = re.sub(pattern,'', data_train['text'].iloc[i], flags=re.MULTILINE)
print data_train['text'][0]
data_train.head()

#remove URL
import re
pattern=r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
for i in range(len(data_train)):
    data_train['text'].iloc[i] = re.sub(pattern,'', data_train['text'].iloc[i], flags=re.MULTILINE)
print data_train.text[0]
data_train.head()

#remove RT
pattern=r'rt @\w+: '
for i in range(len(data_train)):
    data_train['text'].iloc[i] = re.sub(pattern,'', data_train['text'].iloc[i], flags=re.MULTILINE)
print data_train['text'][0]
data_train.head()

#remove @
pattern=r'@\w+ '
for i in range(len(data_train)):
    data_train['text'].iloc[i] = re.sub(pattern,'', data_train['text'].iloc[i], flags=re.MULTILINE)
data_train.head()

#remove bad character
#[^A-Za-z ]
import re
pattern=r'[^A-Za-z ]'
for i in range(len(data_train)):
    data_train['text'].iloc[i] = re.sub(pattern,'', data_train['text'].iloc[i], flags=re.MULTILINE)
print data_train['text'][0]
data_train.head()

# remove punctuation
import string
string.punctuation
remove=string.punctuation
for i in range(len(data_train)):''
sent=data_train['text'].iloc[i]
kd=' '.join(word.strip(remove) for word in sent.split())
data_train['text'].iloc[i]=kd
print data_train.text[0]
data_train.head()

# normalization
import re
import string
import csv

reader = csv.reader(open('corpus/normalisasi.csv', 'r'))
d = {}
for row in reader:
    k,v= row
    d[string.lower(k)] = string.lower(v)
    #print d[k]
pat = re.compile(r"\b(%s)\b" % "|".join(d))
for i in range(len(data_train)):
    text = string.lower(data_train['text'].iloc[i])
    text = pat.sub(lambda m: d.get(m.group()), text)
    #print text
    data_train['text'].iloc[i]=text
print data_train.text[0]
data_train.head(10)

# remove stopwords
import nltk 
from nltk.corpus import stopwords
reader=pd.read_excel('corpus/stopword_id.xls',header=None)
cachedStopWords = set(stopwords.words("english"))
cachedStopWords.update(reader[0][:])
for i in range(len(data_train)):
    sent=data_train['text'].iloc[i]
    kt=" ".join([word for word in sent.split() if word not in cachedStopWords])
    data_train['text'].iloc[i]=kt
print data_train.text[0]
data_train.head(10)
#data_train.to_csv('data_gojek_random_label_3000_after_punctuation.csv',encoding='utf-8')

# stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
for i in range(len(data_train)):
    sent=data_train['text'].iloc[i]
    output = stemmer.stem(sent)
    data_train['text'].iloc[i]=output
print data_train.text[0]
data_train

# remove the duplicate tweet
data_train=data_train[~data_train['text'].duplicated()]
data_train=data_train.reset_index(drop=True)
print len(data_train)
print data_train.text[0]
data_train.head()

data_proses=data_train
data_proses.to_csv('data_gojek_random_label_3060_stemmed_3.csv',encoding='utf-8')