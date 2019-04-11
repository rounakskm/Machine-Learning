#PCA
#Feature scaling is needed if we apply dimentionality reduction
#PCA will be done after data-preprocessing and before fitting the model

#Logistic Regressioncusing PCA

#Importing Libraries
import numpy as np               #used for mathematical calcs
import pandas as pd              #used to get datasets
import matplotlib.pyplot as plt  #used for plotting charts and graphs

#import dataset
dataset= pd.read_csv('Wine.csv')
X= dataset.iloc[:,0:13].values    #Extracting the features from the dataset
y= dataset.iloc[:, 13].values     #Extracting the Lables from the dataset

               
#Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
   
#Feature Scaling for more accurate results in this case
#this puts the feature varibles in the same range so that one does not dominate the others
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #training data needs to be fitted before it is transformed
X_test = sc_X.transform(X_test)
                 
#PCA (Principal component analysis)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#number of extracted features you want with required varience, we choose none as we dont know the varience
#explained_varience variable is used to store the varience ratio
#this will tell us how much varience is covered by the top components
#then we can make the number of n_components to 2/3 as required. In this case we will use 2
#It will consider the top two components that affect the model the most.
#Reset console and run 


#Fittiing logistic regressionmodel to training data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predicting test set results
y_pred = classifier.predict(X_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
#right diagonal is the number of correctt predictions. Left diagonal is number of incorrect predictions
cm = confusion_matrix(y_test, y_pred)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()