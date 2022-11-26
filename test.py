from keras import layers
from keras import Sequential
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import callbacks
import pickle
dataset: np.ndarray = pd.read_csv(r'./WMT.csv')\
['Open'].to_numpy().flatten()
dataset = np.array([i for i in dataset if i > 0.0])
# print(dataset)
size = 365
last = 365
def load_data(size: int=7, last: int=365*2):
    x_train = []
    y_train = []
    for i  in range(round(len(dataset)-(size))):
        x_train += [dataset[i:i+size]]
        y_train += [dataset[i+size]]
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    l  = 10**len(str(int(dataset.flatten().max())))
    x_train = x_train / l
    y_train = y_train / l
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_train[0-abs(last)::]
    x_train = x_train[:0-abs(last)]
    y_test = y_train[0-abs(last)::]
    y_train = y_train[:0-abs(last)]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data(size, last)
# pickle.dump(((x_train, y_train), (x_test, y_test)), open('data_wmt.pickle',
# 'wb'))
loss = 'huber'

model = Sequential((
    layers.LSTM(64, activation='elu', return_sequences=True,
                input_shape=(size, 1)),
    layers.Dropout(0.1),
    layers.LSTM(128, activation='elu', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(256, activation='elu', return_sequences=True,),
    layers.Dropout(0.3),
    layers.LSTM(512, activation='elu', return_sequences=True,),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(1, activation='elu'),
)) # type: ignore
model.compile('nadam', loss=loss, metrics=['acc'], jit_compile=True)
model.summary()
model.load_weights(f'./{loss}_wmt_{size}.h5')

history, predictions = pickle.load(open(f'./{loss}_wmt_{size}.pickle', 'rb'))
plt.title(f'{loss} regressor training')
plt.xlabel('Epochs')
plt.ylabel('Val')
plt.plot(history['acc'], color='green', label='Accuracy')
plt.plot(history['loss'], color='red', label='Loss')
plt.legend()
plt.show()
# predictions: np.ndarray = model.predict(x_test).flatten()
# plt.title(f'{loss}-{size} prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.plot(predictions, color='blue', label='Predicted')
# plt.plot(y_test.flatten(), color='yellow', label='Real values')
# plt.legend()
# plt.show()
# print(len(predictions))
# predictions = []
# for i in x_test.tolist():
#     # print(i, predictions)
#     # print(i)
#     predictions += [model.predict([i])[0]]
# predictions = np.array(predictions).flatten()
# print(len(predictions))
# plt.title(f'{loss} evaluation')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.plot(predictions, color='blue', label='Predicted')
# plt.plot(y_test.flatten(), color='yellow', label='Real values')
# plt.legend()
# plt.show()
# predictions = predictions[:size+1]
# plt.title(f'{loss} evaluation (trimmed)')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.plot(predictions, color='blue', label='Predicted')
# plt.plot(y_test.flatten(), color='yellow', label='Real values')
# plt.legend()
# plt.show()
predictions: np.ndarray = x_test[-1].reshape(1, size, 1)
iters = []
for i in range(size):
    step = model.predict(predictions)
    predictions = (predictions + step)[-size::]
    iters.append(step[0])
predictions = (np.array(iters))[:size]
print(predictions)
plt.title(f'{loss} evaluation (trimmed) 2')
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(predictions, color='blue', label='Predicted')
plt.plot(y_test.flatten(), color='yellow', label='Real values')
plt.legend()
plt.show()
print(len(predictions))