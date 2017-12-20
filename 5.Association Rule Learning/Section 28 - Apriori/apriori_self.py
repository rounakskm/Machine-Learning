#Aprior 

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
#AS in this data set all rows are just transactions registered, the first row must not be
#considered as the title row. Hence use header = none

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#Number of obs in dataset is 7500 , so we take loop range = 7501
#Number of colums in dataset is 20 so j loop range = 20
#Place everything in the append parameter inside square brackets so that it will be treated as a list
#Convert the dataset.values[i,j] into string as Apriori algorithm takes input as string
                           
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])
    
#Trainning Apriori on dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003,  min_confidence = 0.2,  min_lift=3, max_length = 2)

#For min_support we need a product that is baught atleast 2-3 times a day
# 3X7 = 21 times a week as our dataset is weekly transactions list
# support of that product = 21/7500 =0.003 , we take this as minimum support

#min connfidence taken by general rule of hand

#Better lift = better rules, generally min lift of 3 is good

#Vizualizing the apriori results
results = list(rules)

#These rules are already sorted by relevence.
#This relevence is computed by taking lift,confidence and support. Lift being most important