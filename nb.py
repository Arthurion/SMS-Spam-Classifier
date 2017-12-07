"""
Created on Sat Dec  2 17:23:13 2017

@author: Arthurion9
"""

from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# Importing the dataset
data = pd.read_csv('spambase.data').as_matrix()

### Without Kfold
# Training and testing sets
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data[:,0:56], data[:,57], test_size = 0.2)

# Multinomial Naive Bayes
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Classification rate for NB:", model.score(Xtest, Ytest))


### With K-fold
# K-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
score_NB = 0
for train_index, test_index in kf.split(data):
    Xtrain, Xtest = data[train_index,0:56], data[test_index,0:56]
    Ytrain, Ytest = data[train_index,57], data[test_index,57]
    
    # Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(Xtrain, Ytrain)
    score = model.score(Xtest, Ytest)
    score_NB += score
    print("Classification rate for NB:", score)
score_NB /= kf.get_n_splits()
print(score_NB)
