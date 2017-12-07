# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:03:30 2017

@author: Arthurion9
"""

# Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('spambase.data')
X = data.iloc[:, 0:57].values
Y = data.iloc[:, 57].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 57))

# Adding the second hidden layer
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the third hidden layer
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the fourth hidden layer
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the fifth hidden layer
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(Xtrain, Ytrain, batch_size = 10, epochs = 500)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(Xtest)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest, y_pred)
print(cm)
print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))