#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:40:29 2017

@author: javi
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import PorterStemmer, word_tokenize
from numpy.random import dirichlet
import random
### GIBBS SAMPLER


path = "/Users/b.yc0006/Cloud/term3/textmining/"
data = pd.read_table(path+"speech_data_extend.txt",encoding="utf-8")
data = data.loc[data['year']>=1990]
data = data.reset_index()


def data_preparation(data):
    prep_data = data.apply(lambda row: #tokenize
                            word_tokenize(row['speech'].lower()), axis=1)    
    stop_w=set(stopwords.words('english')) #stopwords
    for i in range(len(prep_data)): #non-alphanumeric characters
        prep_data[i] = [w for w in prep_data[i] if w not in stop_w and w.isalpha()] 
    stemmer = PorterStemmer() #Create a stemmer object
    for i in range(len(prep_data)): #Stem the data
        prep_data[i] = [stemmer.stem(elem) for elem in prep_data[i]]
    unique_words = np.unique([word for doc in prep_data for word in doc]).tolist()
    return prep_data, unique_words


prep_data, unique_words = data_preparation(data)

D = len(prep_data) #Number of documents
K = 2 #Number of topics
V = len(unique_words)#Number of unique terms
Z = prep_data.apply(lambda row: random.choices(range(1,(K+1)),k=len(prep_data[row]))) #Z_dn
N = np.zeros((D,K))
M = np.zeros((K,V))
#Initial values (I took Euan's, no idea how to set them)
alpha = 50/K
eta = 200/V

theta = dirichlet([alpha]*K,D)
beta = dirichlet([eta]*K,V)

# Problem: generate Z_dn, a list of dimension d of lists, each one of different length
# (number of terms of each document) with values in k. So: topic allocation of word
# n in document d. To assign a new topic to each entry, we need to match term v 
# in Beta_v (which will be stored in "unique_words") with word n in Z_dn (which 
# will be stored in "prep_data").

### Sample for topic allocation
def sample_topic(Z, theta, beta):
    D = len(Z)
    for d in range(D):
        n = len(Z[d])
        for i in range(n):
            beta_v = beta[unique_words.index(prep_data[d][i])]
            probs = (theta[d,:]*beta_v)/np.sum(theta[d,:]*beta_v)
            Z[d][i] = np.random.multinomial(1, probs).tolist().index(1)+1

sample_topic(Z, theta, beta)

### Sample for theta

def sample_theta(Z,alpha,theta):
    N = Z.apply(lambda row: np.unique(row, return_counts=True)[1])
    D = theta.shape[0]
    for d in range(D):
        theta[d,:] = dirichlet(N[d] + alpha)
    return theta

theta = sample_theta(N,alpha,theta)

### Sample for beta

Z[d]

def sample_beta(M,eta,beta):
    K = beta.shape[1]
    for k in range(K):
        beta[:,k] = dirichlet(M[k,:] + eta)
    return beta
