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

#Tokenize

prep_data = data.apply(lambda row: 
                        nltk.word_tokenize(row['speech'].lower()), axis=1)

#Remove stop words and non-alphanumeric characters

stop_w=set(stopwords.words('english'))


for i in range(len(prep_data)):
    prep_data[i] = [re.sub('[^\\w]','',w) for w in prep_data[i] 
                                    if (re.sub('[^\\w]','',w) not in stop_w) and 
                                    re.sub('[^\\w]','',w)!=''] 
        

#Stem the data

stemmer = PorterStemmer() #Create a stemmer object

for i in range(len(prep_data)):
    prep_data[i] = [stemmer.stem(elem) for elem in prep_data[i]]

#Unique terms

unique_tokens = []

for i in range(len(prep_data)):
    for word in prep_data[i]:
        if word not in unique_tokens:
            unique_tokens.extend(word)





        
