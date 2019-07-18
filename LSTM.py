# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:41:38 2018

@author: liuyd
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

class ML_Model:
    def __init__(self,HVaR,PnL,time_steps,num_of_units,epochs,batch_size,lr):
        self.HVaR=HVaR
        self.PnL=PnL
        self.time_steps=time_steps
        self.X_train,self.X_test,self.X_valid,self.y_train,self.y_test,self.y_valid=self.data_process(HVaR,PnL,self.time_steps)
        self.num_of_units=num_of_units
        self.epochs=epochs
        self.batch_size=batch_size
        self.lr=lr
        self.model=self.model_train(self.num_of_units,self.epochs,self.batch_size,self.lr)
        
    def model_train(self,num_of_units,epochs,batch_size,lr):
        model = Sequential()
        model.add(LSTM(num_of_units,input_shape=(self.time_steps,1),return_sequences=True))
        model.add(Dense(1))
        model.compile(optimizer=keras.optimizers.Adam(lr=lr),loss='mse')
        model.fit(self.X_train,self.y_train,epochs=epochs,validation_data=(self.X_valid,self.y_valid),batch_size=batch_size)
        return model
    
    def evaluate(self,model):
        pred = model.predict(self.X_test)
        var_min=self.kernel_min(self.X_test,bandwidth=0.05,q=0.01)
        var_max=self.kernel_min(self.X_test,bandwidth=0.01,q=30)
        self.y_pred=pred[:,self.time_steps-1,0]*(var_max-var_min)+var_min

    
    def normalize(self,arr_max,arr_min,arr):
        return (arr-arr_min)/(arr_max-arr_min)

    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[:, ::]
    
    def kernel_min(self,data,bandwidth=0.01,q=0.01):
        X=np.array(data).reshape(len(data),1)
        kde=KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(X)
        return np.percentile(kde.sample(1000000),q)
    
    def processData(self,data,lag):
        X,Y = [],[]
        for i in range(lag,len(data)+1):
            term=[]
            #term.extend(data[i:(i+lag),1])
            term.extend(data[(i-lag):i,0])
            X.append(term)
            Y.append(data[(i-lag):i,1])
        return np.array(X),np.array(Y)
    
    def process(self,data,steps):
        rp_min=min(data[:,0])
        rp_max=max(data[:,0])
        #var_min=min(data[:,1])
        #var_max=0#max(data[:,1])

        var_min=self.kernel_min(data[:800,0],bandwidth=0.05,q=0.01)
        var_max=0#self.kernel_min(data[:800,0],bandwidth=0.01,q=30)   
        data[:,0]=self.normalize(rp_max,rp_min,data[:,0])
        data[:,1]=self.normalize(var_max,var_min,data[:,1])

        #X,y = processData(data,steps) 
        X = self.rolling_window(data[:,0], 20)
        y = self.rolling_window(data[:,1], 20) 

        X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
        y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
        X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train, test_size=0.25)
        return(X_train,X_valid,X_test,y_train,y_test,y_valid)
        
        
    def data_process(self,HVaR,PnL,time_steps=20):
        X_train=[]
        X_test=[]
        X_valid=[]
        y_train=[]
        y_test=[]
        y_valid=[]
    
        for i in range(len(PnL)):
            data=np.column_stack((PnL[i][:-5],HVaR[i][5:]))
            X_train_set,X_valid_set,X_test_set,y_train_set,y_test_set,y_valid_set=self.process(data,time_steps) 
            
            X_train.append(X_train_set)
            X_test.append(X_test_set)
            X_valid.append(X_valid_set)
            y_train.append(y_train_set)
            y_test.append(y_test_set)
            y_valid.append(y_valid_set)
            
        
        #unpack
        X_train = np.concatenate(X_train)
        X_test=np.concatenate(X_test)
        X_valid = np.concatenate(X_valid)
        
        y_train = np.concatenate(y_train)
        y_test=np.concatenate(y_test)
        y_valid = np.concatenate(y_valid)
        
        #reshape
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
        y_train = y_train.reshape((y_train.shape[0],y_train.shape[1],1))
        X_valid=X_valid.reshape((X_valid.shape[0],X_valid.shape[1],1))
        y_valid=y_valid.reshape((y_valid.shape[0],y_valid.shape[1],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
        y_test = y_test.reshape((y_test.shape[0],y_test.shape[1],1))
        
        return(X_train,X_test,X_valid,y_train,y_test,y_valid)