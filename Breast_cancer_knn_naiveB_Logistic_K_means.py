#!/usr/bin/env python
# coding: utf-8

# In[30]:


##Knn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
from sklearn.model_selection import train_test_split


# In[31]:


import math
def Distance(i1,i2,length):
    dist1=0
    x=1
    while x<= length:
        dist=(i1[x]-i2[x])
        dist1=dist1+dist*dist
        x=x+1
    return math.sqrt(dist1)
import operator
from collections import defaultdict


# In[32]:


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


# In[33]:


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
#             for i in range(k):
#         count=neighbours[x][-1]+count
        
    


# In[34]:


def Accu(testSet,predictions):
    correct=0
    wrong=0
    for x in range(len(testSet)):
        
        if testSet[x][-1]==predictions[x]:
            correct+=1
        else:
            wrong=wrong+1
        
    return correct/(correct+wrong)


# In[35]:


def main():
  
    split=0.67
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import optimize as op
    from sklearn.model_selection import train_test_split
    df=pd.read_csv('/home/ajit/Downloads/breast_can/data.csv')
    # df.head()
    df=df.drop('Unnamed: 32',axis=1)
    df.dtypes
    plt.figure(figsize=(12,8))
    sns.countplot(df['diagnosis'],palette='RdBu')
    benign,malignant=df['diagnosis'].value_counts()
   
    cols=['radius_worst',
          'texture_worst','perimeter_worst','area_worst',
         'smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
         'symmetry_worst','fractal_dimension_worst']
    df=df.drop(cols,axis=1)
    df.columns
    cols = ['perimeter_mean',
            'perimeter_se', 
            'area_mean', 
            'area_se']
    df = df.drop(cols, axis=1)
    cols = ['concavity_mean',
            'concavity_se', 
            'concave points_mean', 
            'concave points_se']
    df = df.drop(cols, axis=1)
    target=['M','B']
    m=df.shape[0]
    features=12
    n=features
    c=2
    df.columns
    X=np.ones((m,n+1))
    X[:,0]=df['radius_mean'].values
    X[:,1]=df['texture_mean'].values
    X[:,2]=df['smoothness_mean'].values
    X[:,3]=df['compactness_mean'].values
    X[:,4]=df['symmetry_mean'].values
    X[:,5]=df['fractal_dimension_mean'].values
    X[:,6]=df['radius_se'].values
    X[:,7]=df['texture_se'].values
    X[:,8]=df['smoothness_se'].values
    X[:,9]=df['compactness_se'].values
    X[:,10]=df['symmetry_se'].values
    X[:,11]=df['fractal_dimension_se'].values
#     X[:,12]=df['diagnosis'].values
    

    y=df['diagnosis'].values
    for i in range(m):
        if y[i]=='M':
            X[i,12]=1
        else:
            X[i,12]=0
        for j in range(0,n):
            X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])
    X_train,X_test,y_train,y_test=train_test_split(X,X[:,12],test_size=0.2,
                                                random_state=11)
#     print(y_test[1],X[1,12])
    predictions=[]
    k=3
    x=0
    while x <len(X_test):
        
        nbrs=Neighbors(X_train,X_test[x],k)


        result=Response(nbrs)

        predictions.append(result)
        x=x+1

#         print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = Accu(X_test, predictions)
    print(accuracy*100)
    from sklearn.metrics import f1_score
    f1_model1=f1_score(y_test,predictions,average='weighted',labels=np.unique(predictions))
    print(f1_model1)
    from sklearn.metrics import recall_score
    from sklearn.metrics import  precision_score
    recall=recall_score(y_test,predictions, average='macro')
    print(recall)
    precision=precision_score(y_test,predictions, average='weighted')
    print(precision)

    


# In[36]:




main()


# In[10]:


###
#logistic regression on Breast canser dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
from sklearn.model_selection import train_test_split
df=pd.read_csv('/home/ajit/Downloads/breast_can/data.csv')
    # df.head()
df=df.drop('Unnamed: 32',axis=1)
df.dtypes
plt.figure(figsize=(12,8))
sns.countplot(df['diagnosis'],palette='RdBu')
benign,malignant=df['diagnosis'].value_counts()
   
cols=['radius_worst',
        'texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
        'symmetry_worst','fractal_dimension_worst']
df=df.drop(cols,axis=1)
df.columns
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)
target=['M','B']
m=df.shape[0]
features=12
n=features
c=2
df.columns
X=np.ones((m,n+1))
X[:,0]=df['radius_mean'].values
X[:,1]=df['texture_mean'].values
X[:,2]=df['smoothness_mean'].values
X[:,3]=df['compactness_mean'].values
X[:,4]=df['symmetry_mean'].values
X[:,5]=df['fractal_dimension_mean'].values
X[:,6]=df['radius_se'].values
X[:,7]=df['texture_se'].values
X[:,8]=df['smoothness_se'].values
X[:,9]=df['compactness_se'].values
X[:,10]=df['symmetry_se'].values
X[:,11]=df['fractal_dimension_se'].values
#     X[:,12]=df['diagnosis'].values
    

y=df['diagnosis'].values
for i in range(m):
    if y[i]=='M':
        X[i,12]=1
    else:
        X[i,12]=0
    for j in range(0,n):
        X[:,j]=X[:,j]-X[:,j].mean()
#     print(X[:,12])
X_train,X_test,y_train,y_test=train_test_split(X,X[:,12],test_size=0.2,
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


# In[37]:


#Naive bayes using importing the function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize as op
# data_path='/home/ajit/Downloads/'
# train_data = np.loadtxt(data_path + "haberman1.csv", 
#                         delimiter=",")
# train_features=np.asfarray(train_data[:,:3])
# train_labels=np.asfarray(train_data[:,3:])

df=pd.read_csv('/home/ajit/Downloads/breast_can/data.csv',index_col=0)
# df.head()
df=df.drop('Unnamed: 32',axis=1)
# df.dtypes

cols=['radius_worst',
      'texture_worst','perimeter_worst','area_worst',
     'smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
     'symmetry_worst','fractal_dimension_worst']
df=df.drop(cols,axis=1)

cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)

target=['M','B']
m=df.shape[0]
features=12
n=features
k=2
df.columns
X=np.ones((m,n+1))
X[:,1]=df['radius_mean'].values
X[:,2]=df['texture_mean'].values
X[:,2]=df['smoothness_mean'].values
X[:,4]=df['compactness_mean'].values
X[:,5]=df['symmetry_mean'].values
X[:,6]=df['fractal_dimension_mean'].values
X[:,7]=df['radius_se'].values
X[:,8]=df['texture_se'].values
X[:,9]=df['smoothness_se'].values
X[:,10]=df['compactness_se'].values
X[:,11]=df['symmetry_se'].values
X[:,12]=df['fractal_dimension_se'].values

y=df['diagnosis'].values


from sklearn.model_selection import train_test_split
for j in range(0,n):
    X[:,j]=X[:,j]-X[:,j].mean()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
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


#K_means
import csv
import numpy
with open("/home/ajit/Downloads/breast_can/data.csv") as csvfile:
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
            realclass.append(x[16])
            data2.append(numpy.array(x))
            data.append(numpy.array(x[:-1]))
        else:
            for i in row :
                try:
                    x.append(float(i))
                except:
                    x.append(ord(i))
            testclass.append(x[16])
            test.append(numpy.array(x[:-1]))
    realclass=numpy.array(realclass)
    testclass=numpy.array(testclass)
k=int(input())
centroids=data[:k]
clen=len(centroids)
n=len(data)
d={}
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
        avg=numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        for j in d[i]:
            avg=avg+data[j]
        avg=avg/len(d[i])
        centroids[i]=avg
    #print(len(d[0]),centroids)
final={}
for i in range(clen):
    clas={}
    for j in d[i]:
        clas[data2[j][16]]=clas.get(data2[j][16],0)+1;
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

