#!/usr/bin/env python
# coding: utf-8

# In[46]:



#Kmeans

from pandas import read_csv
import numpy
dataset = read_csv('/home/ajit/Downloads/hepatitis_csv.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)
# count the number of NaN values in each column

#K_means
import csv
import numpy
with open(dataset) as csvfile:
    readCSV=csv.reader(csvfile,delimiter=',')
    data2=[]
    data=[]
    realclass=[]
    next(readCSV,None)
    for row in readCSV:
        x=[];
        for i in row:
            try:
                x.append(float(i))
            except:
                x.append(ord(i))
        realclass.append(x[3])
        data2.append(numpy.array(x))
        data.append(numpy.array(x[:-1]))
    realclass=numpy.array(realclass)
k=int(input())
centroids=data[:k]
print(len(centroids))
x=centroids[0]
clen=len(centroids)
n=len(data)
for i in centroids:
    print(numpy.linalg.norm(i-x))
d={}
print(centroids)
for q in range(100):
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
        avg=[0,0,0];
        for j in d[i]:
            avg=avg+data[j]
        avg=avg/len(d[i])
        centroids[i]=avg
    #print(len(d[0]),centroids)
final={}
for i in range(clen):
    clas={}
    for j in d[i]:
        clas[data2[j][3]]=clas.get(data2[j][3],0)+1;
    final[i]=max(clas.items(),key=lambda k: k[1])[0]
print(final)
l=[int(i) for i in input().split()]
print("class is:")
l=numpy.array(l)
temp=500
for i in range(clen):
    dist=numpy.linalg.norm(centroids[i]-l);
    if(dist<temp):
        temp=dist;
        index=i
print(centroids)
print(centroids[index])
print(final[index])
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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import numpy as np
f1_model1=f1_score(realclass,predict,average='weighted',labels=np.unique(predict))
print(f1_model1)
from sklearn.metrics import recall_score
recall=recall_score(realclass, predict, average='macro')
print(recall)
precision=precision_score(realclass, predict, average='weighted')
print(precision)


# In[5]:


from pandas import read_csv
from sklearn.preprocessing import Imputer
import numpy
dataset = read_csv('/home/ajit/Downloads/hepatitis_csv.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
# count the number of NaN values in each column
print(numpy.isnan(transformed_values).sum())


# In[47]:


#logistic_regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pandas import read_csv
import numpy
dataset = read_csv('/home/ajit/Downloads/hepatitis_csv.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)
# count the number of NaN values in each column
# print(dataset.isnull().sum())

dataset=np.asarray(dataset)

train_features=np.asfarray(dataset[:,:19])
train_labels=np.asfarray(dataset[:,19:])
X=np.array(train_features)
Y=np.array(train_labels)
k=2
n=19
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


# In[12]:


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

from pandas import read_csv
import numpy
dataset = read_csv('/home/ajit/Downloads/hepatitis_csv.csv', header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# fill missing values with mean column values
dataset.fillna(dataset.mean(), inplace=True)
# count the number of NaN values in each column
# print(dataset.isnull().sum())

dataset=np.asarray(dataset)

train_features=np.asfarray(dataset[:,:19])
train_labels=np.asfarray(dataset[:,19:])
X=np.array(train_features)
Y=np.array(train_labels)
k=2
n=19
m=X.shape[0]
for i in range(m):
    for j in range(0,n):
        X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,
                                            random_state=11)
# for i in range(m):
#     for j in range(0,n):
#         X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])

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


# In[43]:


###Knn

import math
def Distance(i1,i2,length):
    distance=0
    for x in range(length):
        
        distance+=((i1[x]-i2[x])*(i1[x]-i2[x]))
    return math.sqrt(distance)
import operator
from collections import defaultdict

# d['i'].append(4)
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
def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import optimize as op
    from sklearn.model_selection import train_test_split
    
    from pandas import read_csv
    import numpy
    dataset = read_csv('/home/ajit/Downloads/hepatitis_csv.csv', header=None)
    # mark zero values as missing or NaN
    dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
    # fill missing values with mean column values
    dataset.fillna(dataset.mean(), inplace=True)
    # count the number of NaN values in each column
    # print(dataset.isnull().sum())

    dataset=np.asarray(dataset)

    train_features=np.asfarray(dataset[:,:19])
    train_labels=np.asfarray(dataset[:,19:])
    X=np.array(train_features)
    Y=np.array(train_labels)
    k=4
    n=18
    m=X.shape[0]
    for i in range(m):
        for j in range(0,n):
            X[:,j]=X[:,j]-X[:,j].mean()
    #     print(X[:,12])
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,
                                                random_state=11)
    # for i in range(m):
#     print(y_test[1],X[1,12])
    predictions=[]
    k=3
    for x in range(len(X_test)):
        neighbors=Neighbors(X_train,X_test[x],k)
        
#         print(neighbors)
        result=Response(neighbors)
#         print(result)
        predictions.append(result)
        
#         print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = Accu(X_test, predictions)
    print(accuracy*100)
    from sklearn.metrics import f1_score
    f1_model1=f1_score(y_test,predictions,average='weighted',labels=np.unique(predictions))
    print(f1_model1)
    import numpy as np
#     f1_model1=f1_score(realclass,predict,average='weighted',labels=np.unique(predict))
#     print(f1_model1)
    from sklearn.metrics import recall_score
    recall=recall_score(y_test,predictions, average='macro')
    print(recall)
    precision=precision_score(y_test,predictions, average='weighted')
    print(precision)
        


# In[44]:


main()


# In[ ]:


# target=['M','B']
# m=df.shape[0]
# features=11
# n=features
# k=2
# df.columns
# X=np.ones((m,n))
# X[:,0]=df['radius_mean'].values
# X[:,1]=df['texture_mean'].values
# X[:,2]=df['compactness_mean'].values
# X[:,3]=df['symmetry_mean'].values
# X[:,4]=df['fractal_dimension_mean'].values
# X[:,5]=df['radius_se'].values
# X[:,6]=df['texture_se'].values
# X[:,7]=df['compactness_se'].values
# X[:,8]=df['symmetry_se'].values
# X[:,9]=df['fractal_dimension_se'].values

# y=df['diagnosis'].values
# Y=np.ones((m,1))
# for i in range(m):
#     if y[i]=='M':
#         Y[i,0]=1
#     else:
#         Y[i,0]=0
# # for j in range(0,n):
# #     X[:,j]=X[:,j]-X[:,j].mean()
# X_train,X_test,y_train,y_test=train_test_split(X,Y[:,0],test_size=0.2,
#                                                 random_state=11)

# data=pd.DataFrame()
# data['target']=y_train
# data['radius_mean']=X_train[:,0]
# # data['smoothness_mean']=X_train[:,2]
# data['texture_mean']=X_train[:,1]
# data['compactness_mean']=X_train[:,2]
# data['symmetry_mean']=X_train[:,3]
# data['fractal_dimension_mean']=X_train[:,4]
# data['radius_se']=X_train[:,5]
# # data['smoothness_se']=X_train[:,7]
# data['compactness_se']=X_train[:,7]
# data['symmetry_se']=X_train[:,8]
# data['fractal_dimension_se']=X_train[:,9]
# data['texture_se']=X_train[:,6]
# n_M=data['target'][data['target']==1].count()
# n_B=data['target'][data['target']==0].count()
# total_MB=data['target'].count()
# p_M=n_M/total_MB
# p_B=n_B/total_MB
# print(p_M,p_B)
# data_means=data.groupby('target').mean()
# # data_means
# data_varience=data.groupby('target').var()
# data_varience
# #mean_for_'1'
# mean_rds_m=data_means['radius_mean'][data_varience.index==1].values[0]
# # mean_smt_m=data_means['smoothness_mean'][data_varience.index==1].values[0]
# mean_txt_m=data_means['texture_mean'][data_varience.index==1].values[0]
# mean_cpt_m=data_means['compactness_mean'][data_varience.index==1].values[0]
# mean_sym_m=data_means['symmetry_mean'][data_varience.index==1].values[0]
# mean_frc_m=data_means['fractal_dimension_mean'][data_varience.index==1].values[0]
# mean_rdss_m=data_means['radius_se'][data_varience.index==1].values[0]
# # mean_smts_m=data_means['smoothness_se'][data_varience.index==1].values[0]
# mean_cpts_m=data_means['compactness_se'][data_varience.index==1].values[0]
# mean_syms_m=data_means['symmetry_se'][data_varience.index==1].values[0]
# mean_frcs_m=data_means['fractal_dimension_se'][data_varience.index==1].values[0]
# mean_txts_m=data_means['texture_se'][data_varience.index==1].values[0]


# #varience_for_'1'
# mean_rds_v=data_varience['radius_mean'][data_varience.index==1].values[0]
# # mean_smt_v=data_varience['smoothness_mean'][data_varience.index==1].values[0]
# mean_txt_v=data_varience['texture_mean'][data_varience.index==1].values[0]
# mean_cpt_v=data_varience['compactness_mean'][data_varience.index==1].values[0]
# mean_sym_v=data_varience['symmetry_mean'][data_varience.index==1].values[0]
# mean_frc_v=data_varience['fractal_dimension_mean'][data_varience.index==1].values[0]
# mean_rdss_v=data_varience['radius_se'][data_varience.index==1].values[0]
# # mean_smts_v=data_varience['smoothness_se'][data_varience.index==1].values[0]
# mean_cpts_v=data_varience['compactness_se'][data_varience.index==1].values[0]
# mean_syms_v=data_varience['symmetry_se'][data_varience.index==1].values[0]
# mean_frcs_v=data_varience['fractal_dimension_se'][data_varience.index==1].values[0]
# mean_txts_v=data_varience['texture_se'][data_varience.index==1].values[0]

# ###mean_for_'0'
# mean_rds_mB=data_means['radius_mean'][data_varience.index==0].values[0]
# # mean_smt_mB=data_means['smoothness_mean'][data_varience.index==0].values[0]
# mean_txt_mB=data_means['texture_mean'][data_varience.index==0].values[0]
# mean_cpt_mB=data_means['compactness_mean'][data_varience.index==0].values[0]
# mean_sym_mB=data_means['symmetry_mean'][data_varience.index==0].values[0]
# mean_frc_mB=data_means['fractal_dimension_mean'][data_varience.index==0].values[0]
# mean_rdss_mB=data_means['radius_se'][data_varience.index==0].values[0]
# # mean_smts_mB=data_means['smoothness_se'][data_varience.index==0].values[0]
# mean_cpts_mB=data_means['compactness_se'][data_varience.index==0].values[0]
# mean_syms_mB=data_means['symmetry_se'][data_varience.index==0].values[0]
# mean_frcs_mB=data_means['fractal_dimension_se'][data_varience.index==0].values[0]
# mean_txts_mB=data_means['texture_se'][data_varience.index==0].values[0]

# #####varience_for_'0'
# mean_rds_vB=data_varience['radius_mean'][data_varience.index==0].values[0]
# # mean_smt_vB=data_varience['smoothness_mean'][data_varience.index==0].values[0]
# mean_txt_vB=data_varience['texture_mean'][data_varience.index==0].values[0]
# mean_cpt_vB=data_varience['compactness_mean'][data_varience.index==0].values[0]
# mean_sym_vB=data_varience['symmetry_mean'][data_varience.index==0].values[0]
# mean_frc_vB=data_varience['fractal_dimension_mean'][data_varience.index==0].values[0]
# mean_rdss_vB=data_varience['radius_se'][data_varience.index==0].values[0]
# # mean_smts_vB=data_varience['smoothness_se'][data_varience.index==0].values[0]
# mean_cpts_vB=data_varience['compactness_se'][data_varience.index==0].values[0]
# mean_syms_vB=data_varience['symmetry_se'][data_varience.index==0].values[0]
# mean_frcs_vB=data_varience['fractal_dimension_se'][data_varience.index==0].values[0]
# mean_txts_vB=data_varience['texture_se'][data_varience.index==0].values[0]
# #Finally, we need to create a function to calculate the probability density of each of the terms of the likelihood 
# def p_x_given_y(x,mean_y,varience_y):
#     p=1/(np.sqrt(2*np.pi*varience_y))*np.exp((-(x-mean_y)**2)/(2*varience_y))
# #     print(1/(np.sqrt(2*np.pi*varience_y)))
# #     print(np.exp((-(x-mean_y)**2)/(2*varience_y)))
# #     p = 1/(np.sqrt(2*np.pi*varience_y)) * np.exp((-(x-mean_y)**2)/(2*varience_y))
#     print(p)
# # i=0
# # p_x_given_y(X_test[i,9],mean_syms_m,mean_syms_v)*\
# # p_x_given_y(X_test[i,10],mean_frcs_m,mean_frcs_v)*\
# # p_x_given_y(X_test[i,11],mean_txts_m,mean_txts_v)
# i=0
# M=p_M*p_x_given_y(X_test[i,0],mean_rds_m,mean_rds_v)*\
# p_x_given_y(X_test[i,1],mean_txt_m,mean_txt_v)*\
# p_x_given_y(X_test[i,2],mean_cpt_m,mean_cpt_v)*\
# p_x_given_y(X_test[i,3],mean_sym_m,mean_sym_v)*\
# p_x_given_y(X_test[i,4],mean_frc_m,mean_frc_v)*\
# p_x_given_y(X_test[i,5],mean_rdss_m,mean_rdss_v)*\
# p_x_given_y(X_test[i,6],mean_cpts_m,mean_cpts_v)*\
# p_x_given_y(X_test[i,7],mean_syms_m,mean_syms_v)*\
# p_x_given_y(X_test[i,8],mean_frcs_m,mean_frcs_v)*p_x_given_y(X_test[i,9],mean_txts_m,mean_txts_v)
        
# print(M)
# def Accuracy(X_test,y_test):
#     correct=0
#     wrong=0
#     for i in range(len(y_test)):
#         if i!=158:
#             M=p_M*p_x_given_y(X_test[i,0],mean_rds_m,mean_rds_v)*p_x_given_y(X_test[i,1],mean_txt_m,mean_txt_v)*p_x_given_y(X_test[i,2],mean_cpt_m,mean_cpt_v)*p_x_given_y(X_test[i,3],mean_sym_m,mean_sym_v)*p_x_given_y(X_test[i,4],mean_frc_m,mean_frc_v)*p_x_given_y(X_test[i,5],mean_rdss_m,mean_rdss_v)*p_x_given_y(X_test[i,6],mean_cpts_m,mean_cpts_v)*p_x_given_y(X_test[i,7],mean_syms_m,mean_syms_v)*p_x_given_y(X_test[i,8],mean_frcs_m,mean_frcs_v)*p_x_given_y(X_test[i,9],mean_txts_m,mean_txts_v)
#             B=p_B*p_x_given_y(X_test[i,0],mean_rds_mB,mean_rds_vB)*p_x_given_y(X_test[i,1],mean_txt_mB,mean_txt_vB)*p_x_given_y(X_test[i,2],mean_cpt_mB,mean_cpt_vB)*p_x_given_y(X_test[i,3],mean_sym_mB,mean_sym_vB)*p_x_given_y(X_test[i,4],mean_frc_mB,mean_frc_vB)*p_x_given_y(X_test[i,5],mean_rdss_mB,mean_rdss_vB)*p_x_given_y(X_test[i,6],mean_cpts_mB,mean_cpts_vB)*p_x_given_y(X_test[i,7],mean_syms_mB,mean_syms_vB)*p_x_given_y(X_test[i,8],mean_frcs_mB,mean_frcs_vB)*p_x_given_y(X_test[i,9],mean_txts_mB,mean_txts_vB)
# #         print(p_M)
# #         print(p_B)
#         print(M)

#         if M>B and y_test[i]==0:
#             correct=correct+1
#         elif B>M and y_test[i]==1:
#             correct=correct+1
#         else:
#             wrong=wrong+1
#     return correct/(correct+wrong)
        
            
            

    
#         Accuracy(X_test,y_test)

