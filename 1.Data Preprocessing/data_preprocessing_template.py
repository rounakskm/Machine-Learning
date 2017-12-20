#Data Preprocessing

#Importing Libraries
import numpy as np               #used for mathematical calcs
import pandas as pd              #used to get datasets
import matplotlib.pyplot as plt  #used for plotting charts and graphs

#import dataset
dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values    #Extracting the features from the dataset
y= dataset.iloc[:, 3].values   #Extracting the Lables from the dataset

               
#Splitting the dataset into train and test data
from sklearn.cross_validation import train_test_split
#X_train and y_train contain training features and labels
#X_test and y_test contain testing features and labels
#test_size= 0.2 takes 20% of the dataset as testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
'''#Feature Scaling
#this puts the feature varibles in the same range so that one does not dominate the others
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #training data needs to be fitted before it is transformed
X_test = sc_X.transform(X_test)
'''                             
