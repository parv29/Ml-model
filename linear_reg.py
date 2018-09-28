# Simple Linear Regression

# Importing the libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler 
s_x= StandardScaler()
xtrain=s_x.fit_transform(xtrain)
xtest=s_x.transform(xtest)

#Fitting the model to xtrain and ytrain

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)


#Predicting the values
ypred=regressor.predict(xtest)



#Visualising results

plt.scatter(xtrain,ytrain,color='orange')
plt.plot(xtrain,regressor.predict(xtrain),color='cyan')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


plt.scatter(xtest,ytest,color='orange')
plt.plot(xtrain,regressor.predict(xtrain),color='cyan')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()