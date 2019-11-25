# -*- coding: utf-8 -*-
"""GNB_LBW_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vCh3BvecQrvmHpi3HDUkOwqohG0g8JE1

Gaussian Naive Bayes implementation on Low Birth Weight Data Set
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')
import pandas as pd

"""Take input from the dataset"""

lbw=pd.read_csv('./data/Final.csv')

print(lbw)

"""Drop columns like history which have zero variance"""

X = lbw.drop("reslt", axis=1)
X=X.drop("history",axis=1) #this gives variance zero, so better drop this

y = lbw["reslt"]

"""Normalize Data before its fed into the model"""

import pandas as pd
from sklearn import preprocessing
#normalize
x =X 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X= pd.DataFrame(x_scaled)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=10)
# Good seed values - 10

print("X_train",X_train)

print("y_train",y_train)

"""Calculated Prior Probabilities for the classes"""

def get_count_unique_vals(labels):
  return dict(labels.value_counts())


def prior_prob(labels):
    counts=get_count_unique_vals(labels)
    number_of_instances=labels.count()
    print(counts.items())
    priors={(key,value/number_of_instances) for key, value in counts.items()}
    return priors
    
priors=prior_prob(y_train)
print("priors",priors)

"""Calculate Mean and Variance For all Features"""

import math

def calculate_mean(df):
  return df.mean()
def calculate_std_dev(df):
  return df.std()

def Calculate_Mean_and_Variance(X_train,y_train):
  mean_and_variance_for_class={}
  classes=y_train.unique()
  for everyclass in classes:
      filtered_training_set=X_train[(y_train==everyclass)]
      mean_and_variance=dict()
      for every_attribute in list(X_train.columns.values):
          particular_attribute=filtered_training_set[every_attribute]
          mean_and_variance[every_attribute]=[]
          mean_for_this_attribute=calculate_mean(particular_attribute)
          mean_and_variance[every_attribute].append(mean_for_this_attribute)
          std_dev_for_this_attribute=calculate_std_dev(particular_attribute)
          var_for_this_attribute=math.pow(std_dev_for_this_attribute,2)
          mean_and_variance[every_attribute].append(var_for_this_attribute)
      mean_and_variance_for_class[everyclass]=mean_and_variance
  return mean_and_variance_for_class

"""For every class and attribute, we keep track of mean and variance"""

dictionary=Calculate_Mean_and_Variance(X_train,y_train)

print("variance and mean for every class and attribute",(dictionary))

import operator

"""NOW using PDF Equation  
Given a feature
"""

def calculate_probability(x, mean,variance):
    return (1/(math.sqrt(2*math.pi*variance))) * math.exp(-(math.pow(x-mean,2)/(2*variance)))

def predict(X_test,mean_variance):
    predictions = {}
    for index, row in X_test.iterrows():
        results = {}
        for k, v in priors:
            p = 0 #probability of every feature
            for feature in list(X_test.columns.values):
                prob = calculate_probability(row[feature], mean_variance[
                    k][feature][0], mean_variance[k][feature][1])
                if prob > 0:
                    p += math.log(prob)
            results[k] = math.log(v) + p
        predictions[index] = max(results.items(), key=operator.itemgetter(1))[0] # assign class that has maximum probability
    return predictions

predictions=predict(X_test,dictionary)

print("predictions for data in test set",predictions)

def acc(y_test,prediction):
  count=0
  for ind,row in y_test.iteritems():
    if row == prediction[ind]:
      count+=1
  return count/len(y_test)*100.0

accuracy=acc(y_test,predictions)
print("accuracy :",accuracy)

#done
