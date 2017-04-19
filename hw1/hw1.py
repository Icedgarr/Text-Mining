# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:23:18 2017

@author: roger
"""


import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

path = "/home/roger/Desktop/BGSE/14D010 Text Mining for Social Sciences/Text-Mining"

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
        






        
