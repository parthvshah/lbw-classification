import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

import pandas as pd

data = pd.read_csv(r'../data/data.csv', sep=',')

# data = normalize(data)
parallel_coordinates(data, 'reslt', color=['#1f77b4', '#ff7f0e'])
plt.savefig('classcombinedparallel.jpg')