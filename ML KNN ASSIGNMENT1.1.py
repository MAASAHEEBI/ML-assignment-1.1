#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[13]:


df=pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt', index_col='Unnamed: 0')
df


# In[14]:


df.info()


# In[16]:


df.describe()


# In[17]:


df.head()


# In[18]:


df.tail()


# In[20]:


df.columns


# In[22]:


df.dtypes


# In[27]:


df.shape


# In[30]:


df.isna().sum()


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:





# In[40]:


sns.scatterplot(data=df,x='Price',y='Sqft')


# In[41]:


sns.distplot( x= df['Price'], hist =False)


#The data has some outliers since it is not standard normal distribution 
#So we will take care of outliers after some visualization


# In[42]:


sns.pairplot(data = df)


# This is summary of all plots of all categories with respect to each other


# In[43]:


sns.boxplot(data= df, x= 'Bathroom')


#our data set has 4, 5 bathrooms as outliers so we need not to use them


# In[44]:


sns.barplot(data = df, x= 'Bedroom', y='Price')


#It is clear that Price is high with more number of Beddrooms in a house


# In[45]:


sns.barplot(data = df, x= 'Living.Room', y='Price')


#There is a week insigh that the price is more with more living rooms


# In[46]:


sns.boxplot(data= df, x= 'Floor')


#There are outliers above 15 floors
#we can delete those records as per our convenience because there are chances for a house to have more than 15 floors


# # Manage data for training & testing (20)

# In[48]:


#Let us define the explanatory variable 'X' and Target variable 'Y' for our ML model
X= df.drop('Price', axis=1)
y= df.Price


# In[49]:


X.head()


# In[50]:


y.head()


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[53]:


#Standard normal scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


x_train_scaled = scaler.fit_transform(X_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
x_test = pd.DataFrame(x_test_scaled)


# In[54]:


x_train.head()


# In[55]:


y_train.head()


# # Finding a better value of k (10)

# In[47]:


from sklearn.metrics import mean_squared_error 
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


#To find best K value using root mean square error


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = KNeighborsRegressor(n_neighbors = K)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

    


# In[57]:


#plotting the rmse values against k values also known as elbow function

curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


#therefore K= 5 has lease error and thus it will have more accuracy


# In[58]:


#finally lets check our model accuracy   !
from sklearn.metrics import accuracy_score


# In[59]:


my_model = KNeighborsRegressor(n_neighbors = 5)

my_model.fit(x_train, y_train)  #fit the model

pred=model.predict(x_test) #make prediction on test set
predictions= pd.DataFrame(pred)


# In[ ]:




