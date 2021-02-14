#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np


# In[21]:


#source= https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/
data= pd.read_csv(r"C:\Users\Niharika\Documents\ML,BA & DS Course Resourse\breast-cancer.data", sep= ",", names= ["Class","age","menopause","tumor-size","inv-nodes","node-caps","deg-malig","breast","breast-quad","irradiat"])


# In[22]:


data


# In[23]:


data.head()


# In[24]:


data.isna().sum()


# In[25]:


data.describe()


# In[26]:


data.count()


# In[27]:


import matplotlib as plt
import seaborn as sns


# In[28]:


sns.pairplot(data)


# In[29]:


sns.barplot(x= "breast", y= "deg-malig", data=data)


# In[30]:


sns.barplot(x= "Class", y= "deg-malig", data=data)


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


feature= data.drop("Class", axis=1).values


# In[33]:


target= data["Class"].values


# In[34]:


X_train,X_test,y_train,y_test= train_test_split(feature,target,test_size=0.33, random_state= 42, stratify= target)


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


logmodel= LogisticRegression()


# In[37]:


#works only for numerical datatypes so this will show error
logmodel.fit(X_train, y_train)


# In[38]:


data.dtypes


# In[39]:


data.head()


# In[40]:


hot_encoded= pd.get_dummies(data)


# In[41]:


hot_encoded.head()


# In[42]:


hot_encoded= pd.get_dummies(data,drop_first=True)


# In[43]:


hot_encoded.head()


# In[44]:


feature1= hot_encoded.drop("Class_recurrence-events", axis=1).values


# In[45]:


target1= hot_encoded["Class_recurrence-events"].values


# In[46]:


X_train,X_test,y_train,y_test= train_test_split(feature1,target1,test_size=0.33, random_state= 42, stratify= target1)


# In[47]:


logmodel.fit(X_train, y_train)


# In[48]:


logmodel.score(X_test, y_test)


# In[49]:


prediction= logmodel.predict(X_test)


# In[51]:


from sklearn.metrics import classification_report


# In[52]:


print(classification_report(y_test, prediction))


# In[60]:


logmodel1= LogisticRegression(penalty='l1',solver='liblinear')


# In[61]:


from sklearn.model_selection import GridSearchCV


# In[62]:


grid= GridSearchCV(logmodel1,{'C':[0.0001, 0.001, 0.01, 0.1, 10]})
grid.fit(X_train,y_train)


# In[63]:


print("Best parameter is:", grid.best_params_)


# In[67]:


logmodel2= LogisticRegression(C=0.0001, penalty='l1', solver= 'liblinear')


# In[68]:


logmodel2.fit(X_train,y_train)


# In[69]:


logmodel2.score(X_test, y_test)


# In[70]:


prediction1= logmodel2.predict(X_test)


# In[71]:


print(classification_report(y_test, prediction1))


# In[ ]:




