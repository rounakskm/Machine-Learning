#Support vector regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#SVR does not include feature scaling                                                   
                                                   
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X  = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# Predicting a new result

# As we have scaled the features and labels, we need to scale 6.5 as well in order to get the correct prediction
# For scaling 6.5 we need to take the sc_X transform object created above
# Be careful as transform method takes only array like matrix as input, passing integer will give error
# We make it a matrix with one cell by using np.array 
# Now we need to apply inverse transform as our output prediction must be in real world tearms not scaled
# We use the sc_y transform object here as it is a lable that we are predicting and sc_y was used to scale the labels

y_pred = sc_y.inverse_transform( regressor.predict(sc_X.transform(np.array([[6.5]]))) )

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()