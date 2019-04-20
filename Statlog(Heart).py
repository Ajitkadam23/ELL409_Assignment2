#!/usr/bin/env python
# coding: utf-8

# In[19]:


#K_means
import csv
import numpy
with open("/home/ajit/Downloads/data/heartstatlog.csv") as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    data2=[]
    data=[]
    realclass=[]
    test=[];
    testclass=[]
    next(readCSV,None)
    count=0
    for row in readCSV:
        x=[];
        count+=1
        if(count<=90):
            for i in row :
                try:
                    x.append(float(i))
                except:
                    x.append(ord(i))
            realclass.append(x[13])
            data2.append(numpy.array(x))
            data.append(numpy.array(x[:-1]))
        else:
            for i in row :
                try:
                    x.append(float(i))
                except:
                    x.append(ord(i))
            testclass.append(x[13])
            test.append(numpy.array(x[:-1]))
    realclass=numpy.array(realclass)
    testclass=numpy.array(testclass)
k=int(input())
centroids=data[:k]
clen=len(centroids)
n=len(data)
d={}
for q in range(230):
    for i in range(clen):
        d[i]=[]
    for i in range(n):
        temp=500;
        for j in range(clen):
            dist=numpy.linalg.norm(data[i]-centroids[j])
            if(dist<temp):
                temp=dist;
                index=j;
        d[index].append(i);
    for i in range(clen):
        avg=numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
        for j in d[i]:
            avg=avg+data[j]
        avg=avg/len(d[i])
        centroids[i]=avg
    #print(len(d[0]),centroids)
final={}
for i in range(clen):
    clas={}
    for j in d[i]:
        clas[data2[j][13]]=clas.get(data2[j][13],0)+1;
    print(clas.items())
    if(len(clas.items())!=0):
        final[i]=max(clas.items(),key=lambda k: k[1])[0]
    else:
        final[i]=0
print(final)
predicted=[]
for l in test:
    print("class is:")
    l=numpy.array(l)
    temp=500
    for i in range(clen):
        dist=numpy.linalg.norm(centroids[i]-l);
        if(dist<temp):
            temp=dist;
            index=i
    print(centroids[index])
    
    print(final[index])
    predicted.append(final[index])
predict=[]
for j in data:
    temp=500
    for i in range(clen):
        dist=numpy.linalg.norm(centroids[i]-j);
        if(dist<temp):
            temp=dist;
            index=i
    predict.append(final[index])
predict=numpy.array(predict)
print(numpy.linalg.norm(predict-realclass))
count=0


for i in range(n):
    if(realclass[i]==predict[i]):
        count+=1
print(count/n)
import numpy as np
from sklearn.metrics import f1_score
f1_model1=f1_score(testclass,predicted,average='weighted',labels=np.unique(predicted))
print(f1_model1)

#     f1_model1=f1_score(realclass,predict,average='weighted',labels=np.unique(predict))
#     print(f1_model1)
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
recall=recall_score(testclass,predicted, average='macro')
print(recall)
precision=precision_score(testclass,predicted, average='weighted')
print(precision)


# In[20]:


##logistic
#logistic_regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data_path='/home/ajit/Downloads/data/'
train_data = np.loadtxt(data_path + "heartstatlog.csv", 
                        delimiter=",")
train_features=np.asfarray(train_data[:,:13])
train_labels=np.asfarray(train_data[:,13:])
X=np.array(train_features)
Y=np.array(train_labels)
k=2
n=13
m=X.shape[0]
for i in range(m):
    for j in range(0,n):
        X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,
                                            random_state=11)
def sigmoid(z):
    return 1.0/(1+np.exp(-z))
def CostFun(ang,X,y,lamb=0.2):
    hi=sigmoid(X.dot(ang))
    g=(lamb/(2**len(y)))*np.sum(ang**2)
    return (1/len(y))*(-y.T.dot(np.log(hi))-(1-y).T.dot(np.log(1-hi)))+g
def Gd(ang,X,y,lamb=0.2):
    m=X.shape[0]
    n=X.shape[1]
    ang=ang.reshape((n,1))
    y=y.reshape((m,1))
    h=sigmoid(X.dot(ang))
    r=lamb*ang/m
    
    re=((1/m)*X.T.dot(h-y))+r
    return re
def logReg(X,y,ang):
    result=op.minimize(fun=CostFun,x0=ang,args=(X,y),method='TNC'
                      ,jac=Gd)
    return result.x


all_ang=np.zeros((k,n))
i=0
target=[0,1]
for j  in target:
    tmp_y=np.array(y_train==j,dtype=int)
#     print(tmp_y)
#     print(X_train)
    optang=logReg(X_train,tmp_y,np.zeros((n,1)))
    all_ang[i]=optang
    i+=1
    
P=sigmoid(X_test.dot(all_ang.T))
# print(P)
p=[target[np.argmax(P[i,:])] for i in range(X_test.shape[0])]

print(accuracy_score(y_test,p)*100)
from sklearn.metrics import f1_score
f1_model1=f1_score(y_test,p,average='weighted',labels=np.unique(p))
print(f1_model1)
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
recall=recall_score(y_test,p, average='macro')
print(recall)
precision=precision_score(y_test,p, average='weighted')
print(precision)
        


# In[21]:


#knn

import math
def Distance(i1,i2,length):
    dist1=0
    x=1
    while x<= length:
        dist=(i1[x]-i2[x])
        dist1=dist1+dist*dist
        x=x+1
    return math.sqrt(dist1)

# d['i'].append(4)
import operator
from collections import defaultdict


def Neighbors(trainSet,testInsta,k):
    dst=[]
    i=0
    x=0
    length=len(testInsta)-1
    nbr = []
    while x <len(trainSet):
        dist=Distance(testInsta,trainSet[x],length)
        dst.append((trainSet[x],dist))
        x=x+1
    dst.sort(key=operator.itemgetter(1))
    for x in range(k):
        nbr.append(dst[x][0])
    return nbr
import operator
import operator
def Response(neighbors):
    clases={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]

        if response in clases:
            clases[response]+=1
        else:
            clases[response]=1
    sortclass=sorted(clases.items(),key=operator.itemgetter(1)
                      ,reverse=True)
    return sortclass[0][0]


def Accu(testSet,predictions):
    correct=0
    wrong=0
    for x in range(len(testSet)):
        
        if testSet[x][-1]==predictions[x]:
            correct+=1
        else:
            wrong=wrong+1
        
    return correct/(correct+wrong)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
from sklearn.model_selection import train_test_split
data_path='/home/ajit/Downloads/data/'
train_data = np.loadtxt(data_path + "heartstatlog.csv", 
                         delimiter=",")
train_features=np.asfarray(train_data[:,:14])
train_labels=np.asfarray(train_data[:,13:])
X=np.array(train_features)
Y=np.array(train_labels)
n=13
m=X.shape[0]
for i in range(m):
    for j in range(0,n):
        X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,
                                                random_state=11)
#     print(y_test[1],X[1,12])
predicted=[]
k=3
x=0   
while x <len(X_test):
    neighbors=Neighbors(X_train,X_test[x],k)
        
#         print(neighbors)
    result=Response(neighbors)
#         print(result)
    predicted.append(result)
    x=x+1
        
#         print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = Accu(y_test, predicted)
print(predicted)
print(accuracy*100)
from sklearn.metrics import f1_score
#     f1_model1=f1_score(y_test,predictions,average='weighted',labels=np.unique(predictions))
#     print(f1_model1)
import numpy as np
f1_model1=f1_score(y_test,predicted,average='weighted',labels=np.unique(predicted))
print(f1_model1)
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
recall=recall_score(y_test,predicted, average='macro')
print(recall)
precision=precision_score(y_test,predicted, average='weighted')
print(precision)
        


# In[ ]:





# In[48]:





# In[22]:


#Naive bayes using importing the function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
from sklearn.model_selection import train_test_split
# data_path='/home/ajit/Downloads/'
# train_data = np.loadtxt(data_path + "haberman1.csv", 
#                         delimiter=",")
# train_features=np.asfarray(train_data[:,:3])
# train_labels=np.asfarray(train_data[:,3:])

data_path='/home/ajit/Downloads/data/'
train_data = np.loadtxt(data_path + "heartstatlog.csv", 
                        delimiter=",")
train_features=np.asfarray(train_data[:,:13])
train_labels=np.asfarray(train_data[:,13:])

# print(train_labels)
X=np.array(train_features)
Y=np.array(train_labels)
# print(Y)
# k=2
# n=3
m=X.shape[0]
# for i in range(m):
#     for j in range(0,n):
#         X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,
                                            random_state=11)
# print(train_labels)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)

accuracy=X_test.shape[0],(y_test != y_pred).sum()
from sklearn.metrics import f1_score
f1_model1=f1_score(y_test,y_pred,average='weighted',labels=np.unique(y_pred))
print(f1_model1)
print(accuracy)
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import numpy as np
# f1_model1=f1_score(realclass,predict,average='weighted',labels=np.unique(predict))
# print(f1_model1)
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred, average='macro')
print(recall)
precision=precision_score(y_test, y_pred, average='weighted')
print(precision)

# https://scikit-learn.org/stable/modules/naive_bayes.html


# In[ ]:




