# -*- coding: utf-8 -*-

import pandas 
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("DatasetAggiornato_ulti.csv")

GAMMA = [] #lista di appoggio

X = dataframe.drop(['target'], axis=1)

Z = X['rooms']
for riga in range (0, Z.size):
  GAMMA.append(Z[riga].replace("[","").replace("]","").replace(" ", ""))

DELTA=np.array([])
for riga in range (0, len(GAMMA)):
	DELTA=np.append(DELTA,np.fromstring(GAMMA[riga], dtype=float, sep=','))
  
DELTA=np.reshape(DELTA,(len(GAMMA),10))

#DELTA CONTIENE I NOSTRI VETTORI DI SPOSTAMENTO A FLOAT
X['rooms'] = DELTA.tolist()

for i in range(0,10):
  X['room_'+str(i)] = np.nan

for i in range(0, len(X)):
  rowCellValue = X.iloc[i, 2]
  for j in range(0,10):
    X.iloc[i, j+3] = rowCellValue[j]

X = X.drop(['rooms'], axis=1)

X

Y = dataframe['target']
Y = list(Y)

# definizione del base model
def baseline_model():
	# creazione del modello
	model = Sequential()
	model.add(Dense(12, input_dim=12, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# compilazione del modello
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# training e valutazione del modello
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
results
