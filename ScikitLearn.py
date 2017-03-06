 
# coding: utf-8

# In[3]:

import sklearn 
import pandas as pd
data=pd.read_csv('data/CardClients.csv')
data.head()


# In[6]:

from sklearn.model_selection import train_test_split
temp_data=data.drop(data.index[0]) #把col欄位名稱踢出data
y=temp_data['Y']       #取出target
x=temp_data.drop('Y',axis=1)  #把資料欄位為target（‘y’）的踢除
train,test,train_target,test_target = train_test_split( x,y,test_size=0.3)
print (test.iloc[2])
print (test_target.iloc[2])


# In[8]:

from sklearn import linear_model
from sklearn.metrics import accuracy_score

clf = linear_model.LogisticRegression()
clf=clf.fit(train,train_target)
log_test_target=clf.predict(test)
acc_log=accuracy_score(test_target,log_test_target)
print("logistic regrssion accuracy")
print (acc_log)


# In[9]:

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
neigh.fit(train,train_target)
knear_test_target=neigh.predict(test)
acc_knear=accuracy_score(test_target, knear_test_target)
print("k-NearestNeighbors accuracy")
print (acc_knear)


# In[10]:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
naive_test_target= gnb.fit(train, train_target).predict(test)
acc_naive=accuracy_score(test_target, naive_test_target)
print("Naive Bayes accuracy")
print(acc_naive)


# In[11]:

from sklearn.ensemble import RandomForestClassifier
random_f=RandomForestClassifier()
random_test_target=random_f.fit(train,train_target).predict(test)
acc_random=accuracy_score(test_target, random_test_target)
print("Random Forest accuracy")
print (acc_random)


# In[12]:

from sklearn.svm import SVC
svm_m=SVC()
svm_test_target=svm_m.fit(train,train_target).predict(test)
acc_svm=accuracy_score(test_target, svm_test_target)
print("svm accuracy")
print(acc_svm)


# In[14]:

from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

classifier = OneVsRestClassifier(linear_model.LogisticRegression())
y_score = classifier.fit(train, train_target).decision_function(test)

#i_log_test_target=map(int,log_test_target)
i_test_target=(map(int,test_target))

fpr, tpr, thresholds = roc_curve(i_test_target,y_score)
roc_auc=auc(fpr,tpr)

fig = plt.figure(0)
plt.cla()
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')


# In[16]:

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

i_knear_test_target=map(int,knear_test_target)
print("k-Nearest Neighbors precision")
print (precision_score(i_test_target,i_knear_test_target))
print("k-Nearest Neighbors recall")
print (recall_score(i_test_target,i_knear_test_target))



# In[17]:

from sklearn.metrics import confusion_matrix
print("Naive Bayes Confusion Matrix")
print(confusion_matrix(test_target,naive_test_target ))


# In[ ]:

from sklearn.svm import SVC
from sklearn import grid_search
import matplotlib.pyplot as plt
import numpy as np


parameters = {'C':[0.001,10,1000],'gamma':[1,1000]}
svc_train=train[['X1','X2']]
svr = SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(svc_train,train_target)
ans=pd.DataFrame(clf.grid_scores_)
#print(ans)
label=['A','B','C','D','E','F']
width = 0.2
x1 = np.array(range(6))*0.9+0.5

fig = plt.figure(1)
plt.cla()
plt.bar(x1, ans['mean_validation_score'], width=width)
plt.ylim(0.77,0.78)

plt.xticks(x1+0.1, label)
plt.xlabel('parameter type')
plt.ylabel('mean_validation_score')
plt.savefig('svm2.png')


# In[ ]:

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

index=[]
sse=[]   #sum of square error (平方的)
k_list=x[['X2', 'X3','X4']]

for num in range(10):
    index.append(num+1)
    kmeans = KMeans(n_clusters=num+1).fit(k_list)
    sse.append(kmeans.inertia_)

fig = plt.figure(2)
plt.cla()
plt.plot(index,sse)
plt.xlabel('numbers of clusters')
plt.ylabel('sum of square error')
plt.savefig('kmeans.png')

# In[ ]:

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
k_list=x[['X2', 'X3','X4']]
parameters = {'n_clusters':[1,2,3,4,5,6,7,8,9,10]}
kmeans=KMeans()
clf = GridSearchCV(kmeans, parameters)
result=clf.fit(k_list)
print("best n_clusters parameter")
print(result.best_params_)

# In[ ]:

from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
k_list=x[['X2', 'X3','X4']]
parameters = {'n_clusters':[3,10]}
kmeans=KMeans()
clf = GridSearchCV(kmeans, parameters)
result=clf.fit(k_list)
clf.cv_results_.keys()
ans=pd.DataFrame(clf.cv_results_)
#print(ans)  
print(ans['mean_test_score'])


