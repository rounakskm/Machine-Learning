#Artificial Neural Network

#Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  #used to encode geography
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()  #used to encode gender
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Creating dummy variables for geography encoded column
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
 
#Removing one column from the onehot encoded dummy variable, to avoid dummy-variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  #model_selecion = cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling helps make computations easy in ANN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Building the Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Use relu for hidden layers and sigmoid for output layer 
# Adding the input and first hidden layer
classifier.add(Dense(units = 6 , use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu' , input_dim =11))

# Units = number of output nodes in the hidden layer, which if your not an artist should be computed as avg of input nodes+output nodes. 
# In this case (11+1)/2=6 . Rule applies for all hidden layers.

# Adding another hidden layer
classifier.add(Dense(units = 6 , use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1 , use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))

# Units =1 as our outcome is binary (customer stays or leaves)
# if output layer has more than one outputs, then units = number of categories. And activation = 'softmax' 

# Compiling the ANN - Applying stochastic gradient descent 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Optimizer is the algorithm used to find optimized set of weights, 
# Here we use adam which is an algorithm for stochastic gradient descent.
# Loss function used with output sigmoid is a logarithmic loss function
# If output is only one node then the logarithmic loss function used is binary_crossentropy but if more than one npde in output layer then categorical_crossentropy
# Metrics argument expects a list of matrics hence it is placed inside [].

# Fitting ANN classifier to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# batch_size = number of iterations after which gradient descent will be run
# 1 epoch = 1 full cycle, where the ANN is trained on the whole data set.  

# Predicting the Test set results
y_pred = classifier.predict(X_test) #Returns probability of customer leaving the bank

# We need to change y_pred from probabilities to boolean values in order for the confusion matrix to work
y_pred = (y_pred > 0.5) #Will return True if y_pred>0.5(threshold is 50%) and false if less

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#If accuracy of training set and test set converge then your model is correct.
