# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:13:15 2018

@author: liuyd
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score,fbeta_score,recall_score,precision_score,roc_auc_score,accuracy_score



def find_threshold(y_test,y_pred,method,beta=1,plot=False):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    if(thresholds[0]>1):
        thresholds[0]-=1
    if method=="G":
        sensitivity = tpr
        specificity = (1 - fpr)
        score = np.sqrt(sensitivity*specificity)
    elif method=="F":
        score=np.array([ metrics.fbeta_score(y_test, (y_pred>=k),beta) for k in thresholds])
#        score=np.array([precision_score(y_test, (y_pred>=k)) for k in thresholds])
    if plot:
        plt.figure(2)
        plt.plot(thresholds,score)
        plt.xlabel('Thresholds')
        plt.ylabel('Scores')
        plt.title('Scores with different thresholds')
        plt.show()      
    print("The highest score is %f with threshold at %f" % (np.amax(score),thresholds[np.argmax(score)]) )    
    return thresholds[np.argmax(score)]

def confusion(y_test,y_pred):
    import seaborn as sns
    from sklearn import metrics
    # Define names for the three Iris types
    names = ['Not Default','Default']
    mat = pd.DataFrame(metrics.confusion_matrix(y_pred,y_test), index=names, columns=names )
    print(mat )
    # Display heatmap and add decorations
    hm = sns.heatmap(mat , annot=True, fmt="d")
    hm.axes.set_ylabel("Predicted Value")
    hm.axes.set_xlabel("True Value")
    hm.axes.set_title("")


dtrain = xgb.DMatrix((x_train), label=y_train)
dval = xgb.DMatrix((x_val), label=y_val)
dtest=xgb.DMatrix((x_test), label=y_test)
print (x_train.shape)
#%%
params={
   'booster':'gbtree',
	'objective': 'binary:logistic',
#    'eval_metric': 'map',
	'eval_metric': 'auc',
	'max_depth':8,
	'lambda':0.8,
	'subsample':0.8,
	'colsample_bytree':0.8,
	'min_child_weight':5, 
	'eta': 0.02333,
	'seed':666,
   'silent':1, 
   'step_size':0.023333,
   'nthread':4
      }



evallist  = [(dval,'eval'), (dtrain,'train')]
bst=xgb.train( params, dtrain,400,evallist,early_stopping_rounds=10 )
xgb.plot_importance(bst)



threshold=find_threshold(y_val,bst.predict(dval),"G",plot=False)
threshold_xgb=find_threshold(y_val,bst.predict(dval),"F",1)

threshold_xgb=0.1012#best under f1  
prob_xgb=bst.predict(dtest)
y_pred=(prob_xgb>threshold_xgb)*1

print('Auc:',roc_auc_score(y_test,bst.predict(dtest)))
print('f1 score:',f1_score(y_test,y_pred))
print('recall score:',recall_score(y_test,y_pred))
print('precision score:',precision_score(y_test,y_pred))
print('accuracy score:',accuracy_score(y_test,y_pred))
confusion(y_test,y_pred)
bst.save_model("xgb_tf.model")

plt.scatter(np.array(list(range(len(y_pred)))),bst.predict(dtest))
plt.scatter(np.array(list(range(len(y_pred))))[y_test==1],bst.predict(dtest)[y_test==1],color="r")


thresholds=[x/5000 for x in range(1,2500)]
recall=[recall_score(y_test,(prob_xgb>k)*1) for k in thresholds]
precision=[precision_score(y_test,(prob_xgb>k)*1) for k in thresholds]

plt.plot(thresholds, recall)
plt.plot(thresholds, precision)

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
smt = SMOTE()
X_train=x_train.copy()
Y_train=y_train.copy()
X_train[np.isnan(X_train)] = 0

X_val=x_val.copy()
Y_val=y_val.copy()
X_val[np.isnan(X_val)] = 0

X_train, Y_train = smt.fit_sample(X_train,Y_train )

rf=RandomForestClassifier(n_estimators=800, n_jobs=-1,max_depth=28,oob_score=True, random_state=1,)
rf.fit(X_train, Y_train)


X_test=x_test.copy()
X_test[np.isnan(X_test)] = 0

val_prob_rf=rf.predict_proba(X_val)[:,1]

threshold_rf=find_threshold(Y_val,val_prob_rf,"F",plot=False)
    
rf_pred=rf.predict_proba(X_test)


confusion(y_test,(rf_pred[:,1]>threshold_rf)*1)

print('Auc:',roc_auc_score(y_test,rf_pred[:,1]))
print('f1 score:',f1_score(y_test,(rf_pred[:,1]>threshold_rf)*1))
print('recall score:',recall_score(y_test,(rf_pred[:,1]>threshold_rf)*1))
print('precision score:',precision_score(y_test,(rf_pred[:,1]>threshold_rf)*1))
print('accuracy score:',accuracy_score(y_test,(rf_pred[:,1]>threshold_rf)*1))

