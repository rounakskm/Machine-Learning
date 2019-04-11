#NLP - classifying reviews into positive/negetive

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3) #3 is used to ignore doubble quotes

#Cleaning the text (cumpolsory step in NLP)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] #List to store all cleaned reviews
for i in range(0,1000) : #Loop iterating through all reviews in dataset
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #This object is used to stem each word of the review to its root form
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    #set will only bring out unique words and make processing faster
    #This loop iterates through all the words in that particular review
    #Joining the words of the list made above
    review = ' '.join(review)
    corpus.append(review) #aadding the cleaned rewier into the corpus

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #X is thw sparse matrix
y = dataset.iloc[:,1].values #Dependent variable vector

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
