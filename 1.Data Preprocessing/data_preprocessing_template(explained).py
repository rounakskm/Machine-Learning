#Data Preprocessing

#Importing Libraries
import numpy as np               #used for mathematical calcs
import pandas as pd              #used to get datasets
import matplotlib.pyplot as plt  #used for plotting charts and graphs

#import dataset
dataset= pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values    #Extracting the features from the dataset

y= dataset.iloc[:, 3].values   #Extracting the Lables from the dataset

#Taking care of missing data by filling it with mean values of those columns
from sklearn.preprocessing import Imputer               
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)  
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])             

#Encoding the categorical data into numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()  #Encodes categorycal data into numbers
X[:, 0] = labelencoder_X.fit_transform(X[:,0])

#Encodes the data differently in number of columns as number of categories like a sparse matrix
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()  
#in this example we are hot encoding the first column 
#and as it has 3 categories it is being transformed into 
#3 columns which operate as a sparse matrix

#Now for the labels we dont need to use OneHotEncoder
#as it is the dependent variable our model will know 
#that they are categories, and not to be compared

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  #Encodes y to 0/1 in place of yes/no

#Splitting the dataset into train and test data
from sklearn.cross_validation import train_test_split
#X_train and y_train contain training features and labels
#X_test and y_test contain testing features and labels
#test_size= 0.2 takes 20% of the dataset as testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   

#Feature Scaling
#this puts the feature varibles in the same range so that one does not dominate the others
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #training data needs to be fitted before it is transformed
X_test = sc_X.transform(X_test)                             

#As this is a clasification model and the output lable is just a few categories,
#we dont need to scale them, but if there were many output labels
#we would need to sacle them

 
 
