#Multiple Linear Regression

#Importing Libraries
import numpy as np               #used for mathematical calcs
import pandas as pd              #used to get datasets
import matplotlib.pyplot as plt  #used for plotting charts and graphs

#import dataset
dataset= pd.read_csv('50_Startups.csv')
X= dataset.iloc[:,:-1].values    #Extracting the features from the dataset
y= dataset.iloc[:, 4].values   #Extracting the Lables from the dataset

               
#Encoding the categorical data into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()  #Encodes categorycal data into numbers
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()  

#Avoiding Dummy variable trap
#Python library automatiaclly takes care of this, but still done for reference
X = X[:, 1:] #Removed the first column

#Splitting the data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)      
   


#Fit model to training test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
      

#Predicting using the testing data
y_pred = regressor.predict(X_test)

#Optimizing the model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)  
#used to append a column of ones in the begining of the features matrix. y=b0*x0+b1*x1....bn*xn. These ones make up x0

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [3]]

#So the best model is the one using only the data in the index=3 column

#Modifying the dataset to include only the columns needed

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)      


#Fit model to training test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
      

#Predicting using the testing data
y_pred = regressor.predict(X_test)
