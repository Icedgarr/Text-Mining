# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:32:12 2017

@author: roger
"""
#USE topicmodels.LDA insted of lda.LDA
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
#import lda
from collections import Counter
import scipy.sparse as ssp
import topicmodels as tpm
import matplotlib.pyplot as plt
from tqdm import tqdm


path = "/home/roger/Desktop/BGSE/courses/14D010 Text Mining for Social Sciences/Text-Mining/hw1"

data = pd.read_table(path+"/speech_data_extend.txt", encoding="utf-8")

data = data.loc[data['year']>=1946]
data = data.reset_index()
#Tokenize

def tok_stem_data(data):
    
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
        
    empty_str = [i for i in range(len(prep_data)) if len(prep_data[i]) == 0]        

    prep_data = prep_data.drop(empty_str)
    
    prep_data = prep_data.reset_index()
    del prep_data['index']
    prep_data.columns = ['speech']
    prep_data = pd.Series(prep_data['speech'])
    
    return prep_data

def count_words(prep_data):
    unique_words = np.unique([word for doc in prep_data for word in doc])
    unique_words = dict(zip(unique_words,range(len(unique_words))))
    D = len(prep_data)
    V = len(unique_words)
    unigram_data = [[word for word in doc] for doc in prep_data]
    X = np.zeros((D,V))
    for k in range(D):
        counts = Counter(unigram_data[k])
        for word in set(unigram_data[k]):
            X[k,unique_words[word]] = counts[word]    
    X = ssp.csr_matrix(X.astype(int))
    return X

prep_data=tok_stem_data(data)

unique_words = np.unique([word for doc in prep_data for word in doc])

X = count_words(prep_data)

K, alpha, beta = [2, 5, 10], [25, 10, 5], [0.0241, 0.0241, 0.0241]

K, alpha, beta = [2, 5, 10], [50., 25., 10., 5.], [0.0241, 0.01, 0.5]

#K, alpha, beta = [2, 5], [25.], [0.0241]

perps = {}

for k in K:
    for a in alpha:
        for b in beta:
            
            Col_Gibbs = tpm.LDA.LDAGibbs(prep_data,k)
    
            Col_Gibbs.set_priors(a,b)
#burn_samples are the number of samples that we will throw (we use them 
#order to get convergence)
    
#jump is the number of samples that we will skip before recording a sample
#to use afterwards.
    
#used_samples is the number of samples that will be used to estimate the topics
    
            burn_samples = 0
            jumps = 25
            used_samples = 150

            Col_Gibbs.sample(burn_samples,jumps,used_samples)

            perp_coll = Col_Gibbs.perplexity() #If it is constant, it is a good sign. 

            perps[str((k,a,b))] = perp_coll
                       #Otherwise we need to burn more samples.


Col_Gibbs = tpm.LDA.LDAGibbs(prep_data,50)

Col_Gibbs.alpha    
    
burn_samples = 0
jumps = 25
used_samples = 150

Col_Gibbs.sample(burn_samples,jumps,used_samples)

perp_coll = Col_Gibbs.perplexity()

perps[str((50,1,0.0241))] = perp_coll

mins = [(item,min(perps[item])) for item in perps]



plt.plot(perps['(2, 25.0, 0.0241)'], lw = 1., label = 'K=2')
plt.plot(perps['(5, 10, 0.0241)'], lw = 1., label = 'K=5')
plt.plot(perps['(10, 5, 0.0241)'], lw = 1., label = 'K=10')
plt.plot(perps['(50, 1, 0.0241)'], lw = 1., label = 'K=50')
plt.legend()
plt.ylabel('perplexity')
plt.xlabel('25x samples')
plt.title('perplexity across different K')
plt.savefig('per_coll.png')




perps1 = {}
K = [2,5,10,50]
for k in [2,5,10,50]:
    
    Col_Gibbs = tpm.LDA.LDAGibbs(prep_data,50)
    burn_samples = 0
    jumps = 25
    used_samples = 150

    Col_Gibbs.sample(burn_samples,jumps,used_samples)

    perp_coll = Col_Gibbs.perplexity()

    perps1[str((k,Col_Gibbs.alpha,0.0241))] = perp_coll
    
a = [10,2,5,50]
i = 0
for item in perps1:
    plt.plot(perps1[item], lw = 1., label = 'K='+str(a[i]))
    plt.legend()
    plt.ylabel('perplexity')
    plt.xlabel('25x samples')
    plt.title('perplexity across different K')
    plt.savefig('per_coll.png')
    i+=1

K = 50

Col_Gibbs = tpm.LDA.LDAGibbs(prep_data,K)
alpha =  Col_Gibbs.alpha

burn_samples = 800
jumps = 25
used_samples = 100

Col_Gibbs.sample(burn_samples,jumps,used_samples)

perp_coll = Col_Gibbs.perplexity()

doc_topics = Col_Gibbs.dt #prob of each doc to belong to each topic

Nd = np.array([len(par) for par in prep_data])
Nd = Nd.reshape((len(prep_data),1))
Ndtop = Nd*doc_topics

theta_coll = (Ndtop+alpha)/(Nd+K*alpha)
theta_coll =  pd.DataFrame(theta_coll)
pd.DataFrame.to_csv(theta_coll,path_or_buf='data/theta_collapsed.csv',index=False)


theta_uncoll = pd.read_csv('data/theta_uncollapsed.csv') 
theta_uncoll.drop( 'Unnamed: 0',axis=1,inplace=True)
theta_uncoll = np.array(theta_uncoll)


word_topics = Col_Gibbs.tt #prob of each word to belong to each topic
doc_topics = Col_Gibbs.dt #prob of each doc to belong to each topic


#Col_Gibbs.topic_content(20)

dt = Col_Gibbs.dt_avg() #For each doc, prob of each topic






#K, S, alpha, eta = 2, 1000, 0.1, 0.01
#Col_Gibbs1 = lda.LDA(n_topics=K, n_iter=S, alpha=alpha, eta=eta)
#Col_Gibbs1.fit_transform(X)
#Col_Gibbs1.doc_topic_


