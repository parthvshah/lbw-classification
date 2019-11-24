import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

url0 = 'data/part_0.csv'
url1 = 'data/part_1.csv'

def read_data(class1, class2):
    data_class1 = pd.read_csv(class1)
    data_class2 = pd.read_csv(class2)
    data = data_class1.append(data_class2)
    return data

def normalize(df):
    norm_df = (df-df.min())/(df.max()-df.min())
    return norm_df

def calculate_split(length, percentage):
    return int(length * percentage)

def split_data(data, value):
    split = calculate_split(len(data), 0.10)
    train = data[split:]
    test = data[:split]
    return train, test

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)

def manhattan_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += abs(row1[i] - row2[i])
	return distance

def print_test_split(data):
  count0 = 0
  count1 = 0
  for row in data:
    if(row[-1]==0):
      count0 += 1
    if(row[-1]==1):
      count1 += 1
  # print("In test, class 0:", count0, "class 1:", count1)

def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup:tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def main(seed_val):
  data = read_data(url0, url1)
  data = normalize(data)
  data = data.to_numpy()

  # List of good seed values: 2, 4, 11, 12
  np.random.seed(seed_val)
  np.random.shuffle(data)

  train, test = split_data(data, 0.10)

  print_test_split(test)

  neighbors = 10

  while neighbors!=1:
    count = 0
    for row in test:
      label = predict_classification(train, row, neighbors)
      if(label == row[-1]):
        count += 1
    neighbors -= 1
    print(seed_val, neighbors, count/len(test))