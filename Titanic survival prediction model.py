#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# # Data collection and processing

# In[2]:


df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Titanic survival prediction/train.csv')


# In[3]:


#print the first 5 rows of the dataset
df.head()


# In[4]:


#print the last 5 rows of the dataset
df.tail()


# In[5]:


#shape of the dataset
df.shape


# In[6]:


#Getting some info about the dataset
df.info()


# In[7]:


#checking for any missing values
df.isnull().sum()


# Handling the missing values

# In[8]:


# Cabin column contains less than 25% of the data so we are going to drop the column
df = df.drop(columns = 'Cabin', axis = 1)


# In[9]:


df.isnull().sum()


# In[10]:


# Replacing the missing value in Age column with the mean of the age collumn
mean_val = df['Age'].mean()
df['Age'].fillna(mean_val, inplace = True)


# In[11]:


df.isnull().sum()


# In[12]:


#Replacing the missing value in Embarked column with the mode of the column
mode_val = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_val, inplace = True)


# In[13]:


df.isnull().sum()


# # Data analysis and Visualization

# In[14]:


#Getting some stastical measures about the dataset
df.describe()


# In[15]:


# Checking for the no. of people who survived
df['Survived'].value_counts()


# In[16]:


sns.set_style('darkgrid')
plt.figure(figsize = (7,7))


# In[17]:


# Making a count plot for Survived column
sns.countplot(x = 'Survived', data = df)


# In[18]:


# Making a count plot for Gender column
sns.countplot(x = 'Sex', data = df)


# In[19]:


# No. of survivors based on their gender
sns.countplot(x = 'Sex', hue = 'Survived', data = df)


# In[20]:


# Making a count plot for Pclass column
sns.countplot(x = 'Pclass', data = df)


# In[21]:


# No. of Pclass based on their gender
sns.countplot(x = 'Pclass', hue = 'Survived', data = df)


# Encoding the categorical columns

# In[22]:


objlist = df.select_dtypes('object').columns
objlist = list(objlist)
objlist.pop(0)
objlist.pop(1)
objlist


# In[23]:


encoder = LabelEncoder()


# In[24]:


for col in objlist:
    df[col] = encoder.fit_transform(df[col].astype(str))


# In[25]:


df.head()


# Separating features and labels

# In[26]:


X = df.drop(columns = ['PassengerId','Name', 'Ticket', 'Survived'], axis = 1)
Y = df['Survived']


# # Splitting the data into training and testing data

# In[27]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, stratify = Y, random_state = 2)


# In[28]:


print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


# # Model training
# 
# Logistic regression

# In[29]:


model = LogisticRegression()


# In[30]:


model.fit(x_train, y_train)


# Model evaluation:
# 
# accuracy score

# In[31]:


#on training data
training_predict = model.predict(x_train)

training_accuracy = accuracy_score(y_train, training_predict)

print('TRAINING ACCURACY IS :', training_accuracy)


# In[32]:


#on testing data
testing_predict = model.predict(x_test)

testing_accuracy = accuracy_score(y_test, testing_predict)

print('TESTING ACCURACY IS :', testing_accuracy)

