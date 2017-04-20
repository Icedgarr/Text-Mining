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

path = "/home/roger/Desktop/BGSE/14D010 Text Mining for Social Sciences/Text-Mining/hw1"

data = pd.read_table(path+"/speech_data_extend.txt", encoding="utf-8")

data = data.loc[data['year']>=1900]
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





def stem(tokens,stemmer):
    stems = [stemmer.stem(token) for token in tokens]
    return stems
    
def tokenizer(doc):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(doc.lower())
    tokens = [w for w in tokens if w not in stop_w and w.isalpha()]
    stems = stem(tokens,stemmer)
    return stems

vectorizer = TfidfVectorizer(tokenizer = tokenizer) #Create a tfidf object
tfidfmat = vectorizer.fit_transform(data['speech'])

tfidfmat = tfidfmat.toarray()







        
