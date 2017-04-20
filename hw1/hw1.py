# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:23:18 2017

@author: roger
"""


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

path = "/home/roger/Desktop/BGSE/14D010 Text Mining for Social Sciences/Text-Mining/hw1"

data = pd.read_table(path+"/speech_data_extend.txt", encoding="utf-8")

data = data.loc[data['year']>=1990]
data = data.reset_index()
#Tokenize

prep_data = data.apply(lambda row: 
                        nltk.word_tokenize(row['speech'].lower()), axis=1)

#Remove stop words and non-alphanumeric characters

stop_w=set(stopwords.words('english'))


for i in range(len(prep_data)):
    prep_data[i] = [w for w in prep_data[i] if w not in stop_w and w.isalpha()] 
        

#Stem the data

stemmer = PorterStemmer() #Create a stemmer object

for i in range(len(prep_data)):
    prep_data[i] = [stemmer.stem(elem) for elem in prep_data[i]]


#tf-idf the data

unique_words = np.unique([word for doc in prep_data for word in doc])

num_words_cutoff = round(0.1*len(unique_words))


def stem(tokens,stemmer):
    stems = [stemmer.stem(token) for token in tokens]
    return stems
    
def tokenizer(doc):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(doc.lower())
    tokens = [w for w in tokens if w not in stop_w and w.isalpha()]
    stems = stem(tokens,stemmer)
    return stems

vectorizer = TfidfVectorizer(tokenizer = tokenizer, max_features = num_words_cutoff) #Create a tfidf object
tfidfmat = vectorizer.fit_transform(data['speech'])

tfidfmat = tfidfmat.toarray()


presis = pd.DataFrame(data['president'].unique())
presis['party'] = ['Rep','Dem','Rep','Dem']
presis['color'] = ['W','W','W','B']
presis.columns=['president','party','color']


data = data.join(presis.set_index('president'),on = 'president')

#svd

S = tfidfmat.copy()

S_svd = np.linalg.svd(S)

A = S_svd[0]
V = np.vstack((np.diag(S_svd[1]),np.zeros(shape=(len(S)-len(S_svd[1]),len(S_svd[1])))))
B = S_svd[2]

out = list(range(200,len(S_svd[1])))

V[out] = 0 

S_hat = A.dot(V).dot(B)



#cosine similarities

# Cosine similarity:
def cos_sim (doc1, doc2):
    if np.sum(doc1)==0 or np.sum(doc2)==0:
        sim = 0
    else:
        sim = np.dot(doc1,doc2)/(np.sqrt(np.dot(doc1,doc1))*np.sqrt(np.dot(doc2,doc2)))
    return (sim)



S_Dem = S[data['party']=='Dem']
S_Rep = S[data['party']=='Rep']

S_hat_Dem = S_hat[data['party']=='Dem']
S_hat_Rep = S_hat[data['party']=='Rep']

DemRep = np.mean([cos_sim(S_Dem[i],S_Rep[j]) for i in range(S_Dem.shape[0]) for j in range(S_Rep.shape[0])])
DemRep_hat = np.mean([cos_sim(S_hat_Dem[i],S_hat_Rep[j]) for i in range(S_hat_Dem.shape[0]) for j in range(S_hat_Rep.shape[0])])

Dem2 = np.mean([cos_sim(S_Dem[i],S_Dem[j]) for i in range(S_Dem.shape[0]) for j in range(S_Dem.shape[0])])
Dem2_hat = np.mean([cos_sim(S_hat_Dem[i],S_hat_Dem[j]) for i in range(S_hat_Dem.shape[0]) for j in range(S_hat_Dem.shape[0])])

Rep2 = np.mean([cos_sim(S_Rep[i],S_Rep[j]) for i in range(S_Rep.shape[0]) for j in range(S_Rep.shape[0])])
Rep2_hat = np.mean([cos_sim(S_hat_Rep[i],S_hat_Rep[j]) for i in range(S_hat_Rep.shape[0]) for j in range(S_hat_Rep.shape[0])])




ind = np.arange(3)  # the x locations for the groups
width = 0.35       # the width of the bars

no_hat_values = (DemRep, Dem2, Rep2)
hat_values = (DemRep_hat, Dem2_hat, Rep2_hat)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, no_hat_values, width, color='r')
rects2 = ax.bar(ind + width, hat_values, width, color='b')

ax.set_ylabel('Cosimilarity')
ax.set_title('Cosimilarity between Democrats y Republicans')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('DemRep', 'DemDem', 'RepRep'))




