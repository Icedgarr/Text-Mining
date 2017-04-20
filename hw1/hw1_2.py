import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#path = "/home/roger/Desktop/BGSE/14D010 Text Mining for Social Sciences/Text-Mining/hw1"
path = "/home/chpmoreno//Dropbox/Documents/BGSE/Third_Term/TMSC/homeworks/hw1/github/Text-Mining/hw1"

data = pd.read_table(path+"/speech_data_extend.txt", encoding="utf-8")
dictionary = pd.read_excel(path+"/inquireraugmented.xls")
economic_variables = pd.read_excel(path+"/economicdata.xls").set_index(['year'])
#tf-idf the data

def stem(tokens,stemmer):
    stems = [stemmer.stem(token) for token in tokens]
    return stems

def tokenizer(doc):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(doc.lower())
    tokens = [w for w in tokens if w not in stop_w and w.isalpha()]
    stems = stem(tokens,stemmer)
    return stems

vectorizer = TfidfVectorizer(tokenizer = tokenizer,
                             stop_words = stopwords.words('english')) #Create a tfidf object

tfidfmat = vectorizer.fit_transform(data['speech'])

def tfidfmat_to_tables(tfidfmat):
    desc_vect = tfidfmat.tocsr()
    n_docs = desc_vect.shape[0]
    tfidftables = [{} for _ in range(n_docs)]
    terms = vectorizer.get_feature_names()
    for i, j in zip(*desc_vect.nonzero()):
        tfidftables[i][terms[j]] = tfidfmat[i, j]
    return tfidftables

def tfidfmat_to_df(tfidftable):
    temp = []
    dictlist = []
    for key, value in tfidftable.items():
        temp = [key,value]
        dictlist.append(temp)
    df_temp = pd.DataFrame(dictlist)
    return df_temp

tfidftables = tfidfmat_to_tables(tfidfmat)

def positive_negative_index(tfidfdf, dictionary, positive_words, negative_words, weighted = False):
    if weighted == False:
        index = (sum(tfidfdf.isin(positive_words)[0]) - sum(tfidfdf.isin(negative_words)[0])) / tfidfdf.shape[0]
    else:
        index = (sum(tfidfdf[1][tfidfdf.isin(positive_words)[0]]) - sum(tfidfdf[1][tfidfdf.isin(negative_words)[0]])) / sum(tfidfdf[1])
    return index


def selected_dict_index(tfidfdf, dictionary, words, weighted = False):
    if weighted == False:
        index = sum(tfidfdf.isin(words)[0]) / tfidfdf.shape[0]
    else:
        index = sum(tfidfdf[1][tfidfdf.isin(words)[0]]) / sum(tfidfdf[1])
    return index

stemmer = PorterStemmer()

positive_words = set(stem(dictionary.filter(items = ["Entry", "Positiv"]).dropna(axis = 0)['Entry'].str.lower().astype(str), stemmer))
negative_words = set(stem(dictionary.filter(items = ["Entry", "Negative"]).dropna(axis = 0)['Entry'].str.lower().astype(str), stemmer))

selection = "Hostile"
words = set(stem(dictionary.filter(items = ["Entry", selection]).dropna(axis = 0)['Entry'].str.lower().astype(str), stemmer))

positive_negative_indicator = []
selection_indicator = []
positive_negative_indicator_weighted = []
selection_indicator_weighted = []

for i in range(tfidfmat.shape[0]):
    if len(tfidftables[i]) != 0:
        positive_negative_indicator.append(positive_negative_index(tfidfmat_to_df(tfidftables[i]), dictionary, positive_words, negative_words))
        selection_indicator.append(selected_dict_index(tfidfmat_to_df(tfidftables[i]), dictionary, words))
        positive_negative_indicator_weighted.append(positive_negative_index(tfidfmat_to_df(tfidftables[i]), dictionary, positive_words, negative_words, True))
        selection_indicator_weighted.append(selected_dict_index(tfidfmat_to_df(tfidftables[i]), dictionary, words, True))
    else:
        positive_negative_indicator.append(0)
        selection_indicator.append(0)
        positive_negative_indicator_weighted.append(0)
        selection_indicator_weighted.append(0)

data['pn'] = positive_negative_indicator
data['sel'] = selection_indicator
data['pnw'] = positive_negative_indicator_weighted
data['selw'] = selection_indicator_weighted

data_per_year = pd.concat([data.groupby(['year']).mean(), economic_variables], axis = 1)

data_per_year.corr()

df_norm = (data_per_year - data_per_year.mean()) / (data_per_year.max() - data_per_year.min())

df_norm.dropna().plot();
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()
