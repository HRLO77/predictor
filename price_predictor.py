# the file from which weights in ./weights/ were trained in.
# it is not recommended to use this file (Lots of things are quite redundant and unclean)
from keras import layers
from keras import Sequential
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import callbacks
import pickle

dataset: np.ndarray = pd.read_csv('./WMT.csv')['Open'].to_numpy(dtype=np.uint16).flatten()
size = 90
last = 90
def load_data(size: int=24, last: int=365):
    x_train = []
    y_train = []
    for i  in range(round(len(dataset)-(size)) ):
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
# model.load_weights(f'./{loss}_wmt_{size}.h5')
history: callbacks.History = model.fit(x_train, y_train, epochs=1,
                                       use_multiprocessing=True,
                                       max_queue_size=1000, workers=100,
                                       batch_size=64, )  # batch 16 if required

# model.save_weights(f'./{loss}_wmt_{size}.h5', overwrite=True)

# history, predictions = pickle.load(open(f'./{loss}_wmt_{size}.pickle', 'rb'))
lossy = history.history['loss']
acc = history.history['acc']
predictions: np.ndarray = model.predict(x_test).flatten()
pickle.dump((history.history, predictions), open(f'./{loss}_wmt_{size}.pickle',
                                                 'wb'))

plt.title(f'{loss} regressor training')
plt.xlabel('Epochs')
plt.ylabel('Val')
plt.plot(acc, color='green', label='Accuracy')
plt.plot(lossy, color='red', label='Loss')
plt.legend()
plt.show()
plt.title(f'{loss} evaluation')
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(predictions, color='blue', label='Predicted')
plt.plot(y_test.flatten(), color='yellow', label='Real values')
plt.legend()
plt.show()