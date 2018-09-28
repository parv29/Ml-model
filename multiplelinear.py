# Simple Linear Regression

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=0)



# Creating dummy variables
from sklearn.preprocessing import  OneHotEncoder,LabelEncoder
l=LabelEncoder()
x[:,3]=l.fit_transform(x[:,3])
one=OneHotEncoder(categorical_features=[3])
x=one.fit_transform(x).toarray()
x=x[:,1:]
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)


# Predicting the Test set results
ypred=regressor.predict(xtest)

# Backward Elemination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),axis=1,values=x)
x_opt=x[:,[0,1,3,4,5]]
reg_ols=sm.OLS(endog=y,exog=x_opt).fit()

reg_ols.summary()
x_opt=x[:,[0,1,3,4,5]]
reg_ols=sm.OLS(endog=y,exog=x_opt).fit()

reg_ols.summary()

x_opt=x[:,[0,3,4,5]]
reg_ols=sm.OLS(endog=y,exog=x_opt).fit()

reg_ols.summary()
x_opt=x[:,[0,3]]
reg_ols=sm.OLS(endog=y,exog=x_opt).fit()

reg_ols.summary()


