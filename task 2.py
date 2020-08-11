#!/usr/bin/env python
# coding: utf-8

# # TASK 2 

# # importing libraries
# 

# In[36]:


#importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # read the data set
# 

# In[37]:


#reading the dataset
df=pd.read_csv(r'http://bit.ly/w-data')


# In[38]:


#reading the first five rows of the datasets
df.head()


# In[39]:


#reading the last five datsets
df.tail()


# In[73]:


#information about the dataset
df.describe()


# # shape of dataset

# In[40]:


#no of rows and columns of the datsets
df.shape


# In[41]:


#sum of null data 
df.isnull().sum()


# # Plotting histogram 

# In[42]:


plt.hist('Scores',data=df,bins=10)
plt.show()


# In[43]:


sns.lmplot(x='Hours',y='Scores',data=df)
plt.show()


# In[44]:


#pairplot with seaborn
sns.pairplot(df,hue='Hours',palette='rainbow')


# In[72]:


#jointplot with sns
sns.jointplot('Hours','Scores',data=df,kind='reg')


# # importing sklearn

# In[46]:



from sklearn.model_selection import train_test_split


# In[47]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# # splitting the data set

# In[48]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)


# In[49]:


#importing linear regression 
from sklearn.linear_model import LinearRegression


# In[50]:


Lreg=LinearRegression()


# In[51]:


#fitting the model 
Lreg.fit(train_x,train_y)


# In[52]:


#accuracy on the test model
Lreg.score(train_x,train_y)


# # Predictions for the test dataset

# In[53]:


pred=Lreg.predict(test_x)
pred


# In[54]:


df1=pd.DataFrame({'Actual':test_y,'Predicted': pred})


# In[55]:


df1


# In[56]:


df1.shape


# In[58]:


sns.jointplot('Actual','Predicted',data=df1,kind='reg')


# # Finding out the predicted score with 9.5 hours

# In[75]:


n=float(input())
hours=np.array([n])
hours=hours.reshape(-1,1)
own_pred=Lreg.predict(hours)
print("No of Hours={}".format(hours))
print("predicted score={}".format(own_pred[0]))


# # Finding out the accuracy of our model

# In[65]:


from sklearn import metrics
print('Mean Absolute error :',metrics.mean_absolute_error(test_y,pred))
print('mean squared error :',metrics.mean_squared_error(test_y,pred))
print('root mean squared error :',np.sqrt(metrics.mean_squared_error(test_y,pred)))

