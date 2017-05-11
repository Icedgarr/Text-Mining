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
from collections import Counter
import time
### GIBBS SAMPLER


path = "/home/chpmoreno/Dropbox/Documents/BGSE/Third_Term/TMSC/homeworks/github/Text-Mining/hw1/"
data = pd.read_table(path+"speech_data_extend.txt",encoding="utf-8")
data = data.loc[data['year']>=1946]
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

#theta = dirichlet([alpha]*K,D)
#beta = dirichlet([eta]*K,V)

# Problem: generate Z_dn, a list of dimension d of lists, each one of different length
# (number of terms of each document) with values in k. So: topic allocation of word
# n in document d. To assign a new topic to each entry, we need to match term v
# in Beta_v (which will be stored in "unique_words") with word n in Z_dn (which
# will be stored in "prep_data").


### Auxiliar functions Z sample
def simulate(K, row):
    samples = np.random.multinomial(1,[1/K]*K,len(prep_data[row])).tolist()
    samples_correct = []
    for s in samples:
        samples_correct.append(s.index(1))
    return samples_correct

def N_count(Z_d, K):
    N_count_vector = []
    for k in range(K):
        N_count_vector.append(Z_d.count(k))
    return N_count_vector

### Sample for topic allocation
def sample_topic(Z, theta, beta):
    D = len(Z)
    for d in range(D):
        n = len(Z[d])
        for i in range(n):
            beta_v = beta[unique_words.index(prep_data[d][i])]
            probs = (theta[d,:]*beta_v)/np.sum(theta[d,:]*beta_v)
            Z[d][i] = np.random.multinomial(1, probs).tolist().index(1)
    return Z
#sample_topic(Z, theta, beta)

### Sample for theta
def sample_theta(Z,alpha,theta):
    D,K = theta.shape
    N = np.zeros((D,K))
    for d in range(D):
        #N[d,:] = np.unique(Z[d], return_counts=True)[1]
        N[d,:] = N_count(Z[d], K)
        theta[d,:] = dirichlet(N[d,:] + alpha)
    return theta

### Sample for beta

def sample_beta(Z,prep_data,eta,beta):
    K = beta.shape[1]
    M = np.zeros((K,V))
    #Generate M
    s = [i for sublist in prep_data for i in sublist ]
    z_s = [z for sublist in Z for z in sublist]
    for k in range(K):
        words = [s[i] for i in range(len(s)) if z_s[i] == k]
        counts = Counter(words)
        for v in range(len(unique_words)):
            if unique_words[v] in counts: M[k,v] = counts[unique_words[v]]
    #Generate beta
    for k in range(K):
        beta[:,k] = dirichlet(M[k,:] + eta)
    return beta

#beta = sample_beta(Z,prep_data,eta,beta)

def gibbs_sampler(n_iter,prep_data,alpha,eta,K):
    ## Initialize objects
    D = len(prep_data)
    theta = dirichlet([alpha]*K,D)
    beta = dirichlet([eta]*K,V)
    Z = prep_data.apply(lambda row: simulate(K,row))
    Z_dist = []
    theta_dist = []
    beta_dist = []
    for i in range(n_iter):
        print('Iteration nº:'+ str(i))
        start = time.time()
        Z = sample_topic(Z,theta,beta)
        theta = sample_theta(Z,alpha,theta)
        beta = sample_beta(Z,prep_data,eta,beta)
        Z_dist.append(Z)
        theta_dist.append(theta)
        beta_dist.append(beta)
        print('Duration:'+ str(time.time()-start))
    return Z_dist, theta_dist, beta_dist

### Initial parameters
#D = len(prep_data) #Number of documents
#V = len(unique_words)#Number of unique terms
#Z = prep_data.apply(lambda row: simulate(K,row)) #Z_dn
#N = np.zeros((D,K))
#M = np.zeros((K,V))

#Initial values (reference original paper)
K = 10 #Number of topics
alpha = 50/K
eta = 200/V

Z_2, theta_2, beta_2 = gibbs_sampler(5000,prep_data, alpha, eta, K)
