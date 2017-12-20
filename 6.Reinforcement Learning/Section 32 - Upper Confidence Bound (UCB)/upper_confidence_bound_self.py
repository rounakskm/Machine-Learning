#UCB
#Solving Multi-Armed Bandit problem

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#Importing the datatset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Global variables
N = 10000 #N-> Total number of rounds
d = 10 #d->number of ads
ads_selected = [] #Lists of ads selected at each round
total_reward = 0


#Implementing UCB
#d->number of ads
numbers_of_selections = [0]* d #Vector of size d initialized to 0 
sums_of_rewards = [0]* d

#Computing average reward and upper bound for each ads                  
#n-> each round
for n in range(0,N):
    ad = 0                          #used to keep track of this specific ad
    max_upper_bound = 0             #Computed once each round
    for i in range(0,d):
        #if condition to make sure each ad has been selected atleast once
        if(numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i]) #n+1 because indexes in python starts at 0
            upper_bound = average_reward+delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound :
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)             #append selected ad to list
    numbers_of_selections[ad] = numbers_of_selections[ad]+1        #Update number of selections for each ad
    reward = dataset.values[n,ad] #Getting the result from the simulation dataset
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward           
              
#In the ads_selected vector, if we look at the last rounds then we can see that only one ad has been used.
#This ad is the mostt efficient ad. We get the ads index here. So if ads_selected uses ad index 4 then the ad used is ad number 5
#Also compared to simple random selection our total rewards have almost doubled.


#Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()            
            
            
            
            
            
            
            
            
            
            
            