# this just visualizes some dummy data, price data, and the current splitting algorithm.
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
data: np.ndarray = pd.read_csv('./WMT.csv')['Open'].to_numpy().flatten()
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
size = 2
last = 7
x_train = []
y_train = []
for i in range(round(len(dataset)-(size))):
    x_train += [dataset[i:i+size]]
    y_train += [dataset[i+size]]
    
x_train = np.array(x_train)
y_train = np.array(y_train)
l  = 10**len(str(int(dataset.flatten().max())))
x_train = x_train / l
y_train = y_train / l
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_train[0-abs(last)::]
x_train = x_train[:0-abs(last)]
y_test = y_train[0-abs(last)::]
y_train = y_train[:0-abs(last)]
print(dataset, end='\n\n\n\n')
print(x_train, y_train, sep='\n\n\n')
print('snip')
print(x_test, y_test, sep='\n\n\n')
plt.title('Test stock price data')
plt.xlabel('Price')
plt.ylabel('Days')
plt.plot(data, color='blue')
plt.show()