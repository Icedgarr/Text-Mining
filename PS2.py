import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import PorterStemmer, word_tokenize
from numpy.random import dirichlet
from collections import Counter
from scipy.sparse import csr_matrix
import time

### GIBBS SAMPLER

path = "/home/chpmoreno/Dropbox/Documents/BGSE/Third_Term/TMSC/homeworks/github/Text-Mining/hw1/"
path2 = "/home/chpmoreno/Dropbox/Documents/BGSE/Third_Term/TMSC/homeworks/github/Text-Mining/"
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
    D = len(prep_data)
    V = len(unique_words)
    X = np.zeros((D,V))
    N = 0
    for i in range(D):
        N = N + len(prep_data[i])
        aux_words_d = list(set(prep_data[i]))
        for j in range(len(aux_words_d)):
            X[i,unique_words.index(aux_words_d[j])] = prep_data[i].count(aux_words_d[j])
    X = csr_matrix(X.astype(int))
    return prep_data, unique_words, X, N

# Problem: generate Z_d,n, a list of dimension d of lists, each one of different length
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

def gibbs_sampler(n_iter,prep_data,alpha,eta, K, X, N, prop_perplexity):
    ## Initialize objects
    D = len(prep_data)
    theta = dirichlet([alpha]*K,D)
    beta = dirichlet([eta]*K,V)
    Z = prep_data.apply(lambda row: simulate(K,row))
    Z_dist = []
    theta_dist = []
    beta_dist = []
    perplexity = []
    for i in range(n_iter):
        print('Iteration nº:'+ str(i))
        start = time.time()
        Z = sample_topic(Z,theta,beta)
        theta = sample_theta(Z,alpha,theta)
        beta = sample_beta(Z,prep_data,eta,beta)
        if (i % (round(n_iter * prop_perplexity) + 1)) == 0:
            perplexity.append(np.exp(-np.sum(X.multiply(np.log(theta.dot(beta.T))))/N))
            np.save(path2+"perplexity.npy",perplexity)
        Z_dist.append(Z)
        theta_dist.append(theta)
        beta_dist.append(beta)
        np.save(path2+"theta.npy",theta_dist)
        np.save(path2+"Z_dist.npy",Z_dist)
        np.save(path2+"beta_dist.npy",beta_dist)
        print('Duration:'+ str(time.time()-start))
    return Z_dist, beta_dist, theta_dist, perplexity

### Initial parameters
prep_data, unique_words, X, N = data_preparation(data)

#Initial values (reference original paper)
K = 2 #Number of topics
alpha = 50/K
V = len(unique_words)
eta = 200/V

Z_2, beta_2, theta_2, perplex_2 = gibbs_sampler(100, prep_data, alpha, eta, K, X, N, 0.05)
