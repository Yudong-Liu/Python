# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:47:20 2017

@author: liuyd
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow  as tf

num_units=30       #The number of units in the LSTM cell
input_size=47
output_size=1
lr=0.00055         #learning rate
iteration=100
#——————————————————load data——————————————————————
f=open('data.csv')
df=pd.read_csv(f)     #read data
data=df.iloc[:,1:49].values  


#——————————————————Functions——————————————————————
def get_train_data(batch_size,time_step,train_begin,train_end):
    batch_index=[]
    data_train=data[train_begin:train_end]
    mean=np.mean(data_train,axis=0)
    std=np.std(data_train,axis=0)  
    normalized_train_data=(data_train-mean)/std
    train_x,train_y=[],[]   #train set
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:input_size]
       y=normalized_train_data[i:i+time_step,input_size,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#——————————————————————————————————————————————————
def get_test_data(time_step,test_begin,test_end):
    data_test=data[test_begin:test_end]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  
    size=(len(normalized_test_data)+time_step-1)//time_step  
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]
       y=normalized_test_data[i*time_step:(i+1)*time_step,input_size]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
    return mean,std,test_x,test_y

#————————————————————————————————————————————————————
#——————————————————define variables——————————————————
#the weight and biases of input and output

weights={
         'in':tf.Variable(tf.random_normal([input_size,num_units])),
         'out':tf.Variable(tf.random_normal([num_units,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[num_units,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————define the neural newwork variables——————————————————
def lstm(X):
   
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #transform it into 2 dimension,input the results as hidden states
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,num_units]) #transform it into 3 dimension, as the input of lstm cell 
    cell=tf.nn.rnn_cell.BasicLSTMCell(num_units)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,num_units]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

#————————————————training model————————————————————

def train_lstm(batch_size=60,time_step=30,train_begin=0,train_end=8000):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iteration):     
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print("Number of iterations:",i," loss:",loss_)
        print("model_save: ",saver.save(sess,'model_save2\\modle.ckpt'))
        #save the model under this directory
        print("The train has finished")

#————————————————prediction————————————————————
def predict(time_step,test_begin,test_end):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step,test_begin,test_end)
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #denormalize
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[input_size]+mean[input_size]
        test_predict=np.array(test_predict)*std[input_size]+mean[input_size]
        mape=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  
        mae=np.average(np.abs(test_predict-test_y[:len(test_predict)]))
        mse=np.average((test_predict-test_y[:len(test_predict)])*(test_predict-test_y[:len(test_predict)]))
        print("The MAE of this predict:",mae)
        print("The MSE of this predict:",mse)
        print("The MAPE of this predict:",100*mape,"%")
        #plot
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b',)
        plt.plot(list(range(len(test_y[:len(test_predict)]))), test_y[:len(test_predict)],  color='r')
        plt.show()
#————————————————————————————————————————————————————
train_lstm(batch_size=64,time_step=5,train_begin=0,train_end=8000)
predict(time_step=5,test_begin=8000,test_end=9551)
