#!/usr/bin/env python
# coding: utf-8

# ## Task 1: Regression

# **importing necessary libraries** 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# **Using simple linear regression on the houseprice_data.csv dataset to develop a model to predict the price
# of a house**

# In[5]:


df = pd.read_csv('C:/Users/Hp 2022/Documents/dataset/houseprice_data.csv')


# In[6]:


print(df.head()) 
print(df.info()) 
print(df.corr(),'\n') 


# In[8]:


df.corr()['price'].sort_values(ascending =False)


# **Using simple linear regression with X= 3(sqft_living) , Y = 0(Price)**

# In[7]:


X = df.iloc[:, [3]]
y = df.iloc[:, 0]


# visualisng initial data set 

# In[10]:


feature1 = 'sqft_living'  # Replace with your feature column name
feature2 = 'price'  # Replace with another feature column name
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

# Scatter plot of Feature1 vs Feature2
plt.scatter(df[feature1], df[feature2], s=50, c='blue', alpha=0.7, label=f'{feature1} vs {feature2}')

# Customize the plot
plt.title(f'Scatter Plot: {feature1} vs {feature2}')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
plt.tight_layout()
plt.savefig('initial_scatter_plot.png') 


# In[10]:


# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)
# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: ', regr.coef_)
# The coefficients
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))


# In[11]:


#visualise test set results
fig1, ax1 = plt.subplots()
ax1.scatter(X_test, y_test, color = 'red')
ax1.plot(X_test, regr.predict(X_test), color = 'blue')
ax1.set_title('Price vs sqft_living (Test set)')
ax1.set_xlabel('sqft_living')
ax1.set_ylabel('Price')
fig1.tight_layout()
fig1.savefig('LR_test_plot.png')


# **including many features and see if it improves**

# In[18]:


X = df.iloc[:, [3,9,5,6]].values # inputs
y = df.iloc[:, 0]


# In[19]:


# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)
# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: ', regr.coef_)
# The coefficients
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))


# In[21]:


# Create a pairplot for the testing data
pairplot_test = sns.pairplot(df, vars=['sqft_living', 'grade', 'floors', 'waterfront'], hue='price')
pairplot_test.fig.suptitle("Pairplot for Testing Data", y=1.02)
pairplot_test.savefig('new_test_plot.png')
plt.show()


# **How i could make future improvement**
# we could make improvement by carryng out normilistion and dropping of some unecessary columns 

# In[230]:


from sklearn.preprocessing import MinMaxScaler


# In[231]:


df2 = pd.read_csv('C:/Users/Hp 2022/Documents/dataset/houseprice_data.csv')


# In[232]:


columns_to_drop = ['lat','long', 'waterfront', 'view']


# In[233]:


df2 = df2.drop(columns=columns_to_drop)


# In[234]:


features_to_normalize = ['sqft_basement', 'yr_renovated']
scaler = MinMaxScaler()


# In[235]:


df2[features_to_normalize] = scaler.fit_transform(df2[features_to_normalize])


# In[236]:


df2


# In[240]:


X = df2.iloc[:,0:15].values
y = df2.iloc[:, 0]


# In[241]:


# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)
# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)
# The coefficients
print('Coefficients: ', regr.coef_)
# The coefficients
print('Intercept: ', regr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))


# In[ ]:




