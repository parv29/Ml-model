# Simple Linear Regression

# Importing the libraries



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values

y=dataset.iloc[:, 2].values

# Fitting decision tree model to the Training set

from sklearn.ensemble import RandomForestRegressor

regressor= RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

# prediciting result

ypred=regressor.predict(X_grid)

# Visualising result
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
y_pred=regressor.predict(6.5)
plt.scatter(x,y,color='orange')
plt.plot(X_grid,regressor.predict(X_grid),color='cyan')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()
