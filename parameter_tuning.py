# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:27:36 2018

@author: liuyd
"""

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.model_selection import KFold
from keras.layers import Dropout
from keras.constraints import maxnorm
import itertools
import time



#path = "C:/Users/liuyd/Desktop/true var new/"
#os.chdir(path)

#df=pd.read_csv('simulation_data.csv')     #read data
df2=np.genfromtxt('simulation_data.csv', delimiter=',')
#data=df.iloc[:,[4,5]].values
data = df2[1:, [4,5]]

rp_min=min(data[:600,0])
rp_max=max(data[:600,0])
var_min=min(data[:600,0])
var_max=max(data[:600,0])

def normalize(arr_max,arr_min,arr):
    return (arr-arr_min)/(arr_max-arr_min)

data[:,0]=normalize(rp_max,rp_min,data[:,0])
data[:,1]=normalize(var_max,var_min,data[:,1])

#Create a function to process the data into 7 day look back slices
def processData(data,lag):
    X,Y = [],[]
    for i in range(lag,len(data)+1):
        term=[]
        #term.extend(data[i:(i+lag),1])
        term.extend(data[(i-lag):i,0])
        X.append(term)
        Y.append(data[(i-lag):i,1])
    return np.array(X),np.array(Y)


time_steps=20
X,y = processData(data,time_steps) 
#Reshape data for (Sample,Timestep,Features) 
X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
#X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train, test_size=0.25)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
y_train = y_train.reshape((y_train.shape[0],y_train.shape[1],1))
#X_valid=X_valid.reshape((X_valid.shape[0],X_valid.shape[1],1))
#y_valid=y_valid.reshape((y_valid.shape[0],y_valid.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
y_test = y_test.reshape((y_test.shape[0],y_test.shape[1],1))

def create_model(learn_rate,num_units,dropout_rate,weight_constraint):
    model = Sequential()
    model.add(LSTM(num_units,input_shape=(X.shape[1],1),return_sequences=True))
    model.add(Dense(1,kernel_initializer=keras.initializers.random_normal(stddev=0.001),activation='sigmoid', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate, beta_1=0.9,beta_2=0.999,decay=0.0),loss='mse')
    return model

learn_rate=[ 0.05]
num_units=[150]
dropout_rate=[0.0]
weight_constraint=[1,2]   

parameters=list(itertools.product(*[learn_rate,num_units,dropout_rate,weight_constraint]))


kfold = KFold(n_splits=4, shuffle=True, random_state=1)
res=[]
for i in range(len(parameters)):   
    t1=time.clock()
    print('The ',i,' search')
    (learn_rate,num_units,dropout_rate,weight_constraint)=parameters[i]
    model=create_model(learn_rate,num_units,dropout_rate,weight_constraint)        
    scores=[]
    for (train, valid) in kfold.split(X_train, y_train):
        model.fit(X_train[train],y_train[train],epochs=100,batch_size=133,shuffle=False,verbose=0)
        scores.append(model.evaluate(X_train[valid], y_train[valid]))
    res.append(np.mean(scores))
    print('The ',i,' mean loss is ',res[i])
    print('The ',i,' search finish')
    del model
    t2=time.clock()
    print('The ',i,' finished, costing ',t2-t1,'s\n' )

