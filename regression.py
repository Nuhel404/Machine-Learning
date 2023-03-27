#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV


# In[66]:


test_df = pd.read_csv("housing_coursework_test (1).csv")
train_df = pd.read_csv("housing_coursework_train (2).csv")


# In[67]:


train_df.head(5)


# In[68]:


test_df


# In[69]:


train_df=train_df.copy()
train_df.drop(["latitude", "longitude"], axis = 1, inplace=True)
display(train_df)


# In[70]:


train_df.info()


# In[71]:


train_df.dropna(inplace=True)


# In[72]:


train_df.info()


# In[73]:


test_df=test_df.copy()
test_df.drop(["latitude", "longitude"], axis = 1, inplace=True)
display(test_df)


# In[74]:


test_df.info()


# In[75]:


test_df.dropna(inplace=True)


# In[76]:


test_df.info()


# In[77]:


import re

# create a sample DataFrame
df = pd.DataFrame({'A': ['123', '456', 'abc123', 'def456']})

# define a regular expression to match only numeric characters
pattern = re.compile(r'\d+')

# apply the regular expression to the DataFrame column
df['A'] = df['A'].apply(lambda x: pattern.findall(x))

# print the updated DataFrame
print(df)


# In[78]:


train_df = train_df.apply(pd.to_numeric, errors='coerce')
test_df = test_df.apply(pd.to_numeric, errors='coerce')


# In[79]:


train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


# In[80]:


X_train = train_df.drop(columns=['median_house_value'])
y_train = train_df['median_house_value']





# In[81]:


model = LinearRegression()


# In[82]:


model.fit(X_train, y_train)


# In[83]:


X_test = test_df.drop('median_house_value', axis=1)
y_test = test_df['median_house_value']


# In[84]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error:', mse)


# In[108]:


r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)


# In[88]:


#Ridge regression model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[89]:


#create the ridge regression model
ridge = Ridge()


# In[90]:


params = {'alpha': np.logspace(-5, 2, 8)}
grid_search = GridSearchCV(ridge, params, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']


# In[91]:


ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)


# In[104]:


y_pred = ridge.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[105]:


print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R-squared: ", r2)
print("Mean Absolute Error: ", mae)


# In[94]:


#Lasso Regression model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[97]:


lasso = Lasso()


# In[98]:


params = {'alpha': np.logspace(-5, 2, 8)}
grid_search = GridSearchCV(lasso, params, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_alpha = grid_search.best_params_['alpha']


# In[99]:


lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train_scaled, y_train)


# In[106]:


y_pred = lasso.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


# In[107]:


print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)
print("R-squared: ", r2)
print("Mean Absolute Error: ", mae)


# In[ ]:




