
# Simple Linear Regression

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Fitting polynomial  Regression to the Training set

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
x_pol=poly.fit_transform(x)
poly.fit(x_pol,y)

#fitting to linar regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_pol,y)

#predicting values
ypre=regressor.predict(poly.fit_transform(6.5))


# visualising Result
plt.scatter(x,y,color='orange')
plt.plot(x,regressor.predict(poly.fit_transform(x)),color='cyan')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()