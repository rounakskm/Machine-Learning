#Polynomial Regression


#Importing Libraries
import numpy as np               #used for mathematical calcs
import pandas as pd              #used to get datasets
import matplotlib.pyplot as plt  #used for plotting charts and graphs

#import dataset
dataset= pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:2].values    #Extracting the features from the dataset
y= dataset.iloc[:,2].values   #Extracting the Lables from the dataset

#Fit using linear regression
from sklearn.linear_model import LinearRegression  
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fit using polynomial regression             
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree=4) #Drgree can be increased for better results,2 by default
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Vizualizing linear regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Vizualizing Polynomial regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue') #The matrix of features must be in polynomial form not normal
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#Predicting the salary with Linear Regression
lin_reg.predict(6.5)

#Predicting the salary with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))               