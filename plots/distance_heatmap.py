import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import random
import math
import pandas as pd

url0 = '../data/lbw0.csv'
url1 = '../data/lbw1.csv'

def read_data(class1, class2):
    data_class1 = pd.read_csv(class1)
    data_class2 = pd.read_csv(class2)
    data = data_class1.append(data_class2)
    return data

def normalize(df):
    norm_df = (df-df.min())/(df.max()-df.min())
    return norm_df

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return math.sqrt(distance)

data = read_data(url0, url1)
data = normalize(data)
data = data.to_numpy()

map_data = []
for row1 in data:
    row_data = []
    for row2 in data:
        # print(euclidean_distance(row1, row2), end=" ")
        row_data.append(euclidean_distance(row1, row2))
    # print("")
    map_data.append(row_data)

# ax = sns.heatmap(map_data, linewidth=0.5)
# plt.show()

mask = np.zeros_like(map_data)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(map_data, mask=mask, vmax=.3, square=True)
    plt.show()