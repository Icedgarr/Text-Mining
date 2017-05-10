# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:45:10 2017

@author: roger
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# Question 3
# Select only documents which appear after 1945
X = count_words(prep_data)
# Create a new variable with only presidents and years
parties = data.loc[:,['president', 'year']]
parties = parties.reset_index()
# Create a new variable with 1 when presidents are Democrats, 0 when they are Republicans
zero_len = pd.Series(np.zeros(len(parties.index)))
parties['parties'] = zero_len
parties['parties'] = (parties.president == "Truman") | (parties.president == "Kennedy") | (parties.president == "Johnson") | (parties.president == "Carter") | (parties.president == "Clinton") | (parties.president == "Obama")
parties.parties = list(map(lambda x: 1 if x else 0, parties.parties))
parties = parties.iloc[:][parties.year >= 1945]
# Set the parties variable as y
y = parties.parties
# Split the X and y variables into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y, test_size = 0.3, random_state = 42)
# Create the logistic regression estimator with an l1 loss parameter
log_reg = LogisticRegression(penalty = "l1")
# Set some parameters to tune over
c_space = [1.3, 1.5, 1.7]
parameters = {'C': c_space}
# Create a cross-validation estimator
cv = GridSearchCV(log_reg, parameters)
# Fit the model, predict and report the accuracy and the best parameter
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
## Run a logistic regression using the topics instead of the document-term matrix
X = count_words(prep_data)
K, S, alpha, eta = 2, 1000, 0.1, 0.01
Col_Gibbs = lda.LDA(n_topics=K, n_iter=S, alpha=alpha, eta=eta)
X = Col_Gibbs.fit_transform(X)
# Split the X and y variables into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 59)
# Create the logistic regression estimator
log_reg = LogisticRegression("l2")
# Set some parameters to tune over
c_space = [0.5, 2, 5]
parameters = {'C': c_space}
# Create a cross-validation estimator
cv = GridSearchCV(log_reg, parameters)
# Fit the model, predict and report the accuracy and the best parameter
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))