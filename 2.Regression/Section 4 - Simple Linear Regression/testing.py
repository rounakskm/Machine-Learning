#Simple linear regression Population vs Profit

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Population_profit.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
          
#Splitting the dataset into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)        


#Fitting the regressor to the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the result to testing data
y_pred = regressor.predict(X_test)

#Visualizing model aggainst training data
