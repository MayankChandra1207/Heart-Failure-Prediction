#!/usr/bin/env python
# coding: utf-8

# In[130]:


# importing basic libraries for data import and visualization
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[131]:


# importing data set
df=heart_failure=pd.read_csv('project.csv')


# In[132]:


df


# In[133]:


# Showing all Columns


# In[134]:


df.columns


# In[135]:


#Visualization of various parameters using Seaborn


# In[136]:


sns.histplot(df.age)


# In[137]:


sns.histplot(df.anaemia)


# In[138]:


sns.histplot(df.creatinine_phosphokinase)


# In[139]:


sns.histplot(df.diabetes)


# In[140]:


sns.histplot(df.ejection_fraction)


# In[141]:


sns.histplot(df.high_blood_pressure)


# In[142]:


sns.countplot(df.sex)


# In[143]:


#Analysing data using Pandas


# In[144]:


df.head()


# In[145]:


df.tail()


# In[146]:


df.shape


# In[147]:


df.describe()


# In[148]:


df.corr()


# In[149]:


df.info()


# In[150]:


# splitting the data into 20% Test and 80% Training


# In[151]:


x = df.drop(["DEATH_EVENT"],axis=1)
y = df['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=2)


# In[152]:


x_train


# In[153]:


x_test


# In[154]:


y_train


# In[155]:


y_test


# In[156]:


#Importing sklearn library 
#Applying LogisticRegression


# In[157]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[158]:


logmodel = LogisticRegression()


# In[159]:


logmodel.fit(x_train,y_train)


# In[160]:


prediction=logmodel.predict(x_test)


# In[161]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[162]:


confusion_matrix(y_test,prediction)


# In[163]:


accuracy_score(y_test,prediction)


# In[164]:


log_score=accuracy_score(y_test,prediction)*100
log_score


# In[165]:


# Applying KNN Algorithm


# In[166]:


# Importing Required Libraries


# In[167]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[168]:


sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# In[169]:


knn=KNeighborsClassifier(n_neighbors=4)


# In[170]:


knn.fit(x_train,y_train)


# In[171]:


knn_score=knn.score(x_test,y_test)*100
knn_score


# # Comparing Both The Models

# In[172]:


models = ['Logistic Regression' , 'KNN Algorithm']
model_data = [log_score , knn_score]
cols = ["accuracy_score"]
compare=pd.DataFrame(data=model_data , index= models , columns= cols)
compare.sort_values(ascending= False , by = ['accuracy_score'])


# In[173]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(x_test)[:,1])
rf_roc_auc = roc_auc_score(y_test, knn.predict(x_test))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, knn.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='KNN (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# # Confusion Matrix of Both Models

# In[182]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
logmodel_y_pred = logmodel.predict(x_test)
logmodel_cm = metrics.confusion_matrix(logmodel_y_pred, y_test)
sns.heatmap(logmodel_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')


# In[181]:


knn_y_pred = knn.predict(x_test)
knn_cm = metrics.confusion_matrix(knn_y_pred, y_test)
sns.heatmap(knn_cm, annot=True, fmt='.2f')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('KNN')


# In[ ]:




