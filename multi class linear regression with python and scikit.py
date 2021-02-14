#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


#source= htts://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset#


# In[17]:


data= pd.read_csv(r"C:\Users\Niharika\Documents\ML,BA & DS Course Resourse\Bike-Sharing-Dataset\day.csv")


# In[18]:


data


# In[19]:


data.head()


# In[20]:


df= pd.ExcelFile(r"C:\Users\Niharika\Documents\ML,BA & DS Course Resourse\AirQualityUCI\AirQualityUCI.xlsx")


# In[21]:


df.sheet_names


# In[22]:


airquality= df.parse("AirQualityUCI")


# In[23]:


airquality.head()


# In[24]:


airquality.columns


# In[25]:


data.columns


# In[26]:


data.isna().sum()


# In[27]:


data.count()


# In[28]:


data.describe()


# In[31]:


sns.pairplot(data)


# In[32]:


data_1= data[['season','holiday','weekday','workingday','weathersit','temp','windspeed','casual','registered','cnt']]


# In[33]:


sns.pairplot(data_1)


# In[34]:


sns.lineplot("cnt","workingday", data= data_1)


# In[35]:


sns.lineplot("cnt","weekday", data= data_1)


# In[36]:


data_2 =  data[['temp','windspeed','casual','registered','cnt']]


# In[37]:


sns.pairplot(data_2)


# In[38]:


data_2.corr()


# In[39]:


sns.heatmap(data_2.corr())


# In[40]:


sns.barplot(x="cnt",y= "registered", data= data_2)


# In[41]:


sns.distplot(data_2['cnt'])


# In[42]:


sns.distplot(data_2['registered'])


# In[43]:


feature= data_2['registered'].values
target= data_2['cnt'].values


# In[45]:


sns.scatterplot(x=target, y=feature)


# In[47]:


from sklearn.linear_model import LinearRegression


# In[48]:


linear_mod= LinearRegression()


# In[50]:


#this will show error,reshape is required
linear_mod.fit(feature,target)


# In[51]:


feature= feature.reshape(-1,1)
target= target.reshape(-1,1)


# In[53]:


linear_mod.fit(feature, target)


# In[64]:


x_lim= np.linspace(min(feature),max(feature)).reshape(-1, 1)
plt.scatter(feature,target)
plt.xlabel('cnt')
plt.ylabel('registered')
plt.title('cnt vs. registered')
plt.plot(x_lim, linear_mod.predict(x_lim), color = 'red')
plt.show()


# In[66]:


data_2.columns


# In[68]:


X= data_2[['temp','windspeed','casual','registered']]
Y= data_2["cnt"]


# In[70]:


from sklearn.model_selection import train_test_split


# In[73]:


X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.33, random_state= 42)


# In[74]:


X_train.head()


# In[75]:


X_test.head()


# In[76]:


y_train.head()


# In[77]:


y_test.head()


# In[78]:


linear_mod.fit(X_train,y_train)


# In[80]:


print(linear_mod.intercept_)


# In[81]:


linear_mod.coef_


# In[82]:


lm_coef= pd.DataFrame(linear_mod.coef_,X.columns, columns=['Coefficients'])


# In[83]:


lm_coef


# In[84]:


predict= linear_mod.predict(X_test)


# In[85]:


predict


# In[87]:


y_test


# In[88]:


plt.scatter(y_test,predict)


# In[89]:


sns.distplot(y_test-predict)


# In[90]:


from sklearn import metrics


# In[91]:


metrics.mean_absolute_error(y_test, predict)


# In[92]:


metrics.mean_squared_error(y_test, predict)


# In[93]:


np.sqrt(metrics.mean_squared_log_error(y_test, predict))


# In[94]:


predict1= linear_mod.predict(X_train)


# In[95]:


predict1


# In[ ]:




