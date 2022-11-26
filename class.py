import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import layers
from keras import Sequential
import numpy as np
import pandas as pd
from keras import losses
from matplotlib import pyplot as plt
from keras import callbacks
from keras import optimizers
import pickle as pk
import typing
from keras.models import load_model


class predictor:
    '''An LSTM tensorflow.keras.Sequential non-linear (If run with default activation) regression model that can predict things such as weather, stock prices, glucose graphs etc.
    There are 8 pre-trained testing weights files that were fitted on 39 years of WMT stock price data, split into respective intervals of 7, 14, 24, 30, 60, 180 and 365 days each.
    This class is meant to be a simple abstraction on small parts of the pandas, numpy, and keras libraries. (Note that leap years are ignored and taken as regular years)'''
    DIV_CONST = 1000

    def reset_model(self):
        '''Resets the model to the default model with arguments provided on
        instantiation.'''
        model = Sequential((
            layers.LSTM(64, activation=self.activation, return_sequences=True,
                        input_shape=(self.size, 1)),
            layers.Dropout(0.1),
            layers.LSTM(128, activation=self.activation, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(256, activation=self.activation, return_sequences=True,),
            layers.Dropout(0.3),
            layers.LSTM(512, activation=self.activation, return_sequences=True,),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(1, activation=self.activation),
        )) # type: ignore
        model.compile(self.optimizer, loss=self.loss, metrics=['acc'],
                      jit_compile=True, )
        self.model: Sequential = model
    
    @classmethod
    def _load_data(cls, dataset: str | pd.Series | pd.DataFrame | np.ndarray, last: int=1, column: str | None='open', dtype: np.dtype=np.float16, size: int=24, div_const: bool=True):
        '''Runs an experimental prediction algorithm on the dataset given.
        :param cls: The predictor class.
        :param dataset: An iterable, pandas Series/DataFrame, or a string to a filepath to be loaded via-pandas.
        :param last: The last number of dataset entries for testing/evaluation data. Set to 1 for no evalution data returned.
        :param dtype: A np.dtype for the dtype of the loaded dataset.
        :param column: A string. If dataset is a pandas Series/DataFrame or loaded from a filepath via-pandas, the column is selected for the data.
        :param size: The last number of items to predict from the dataset
        :param div_const: Whether or not to divide all data in the dataset by the class DIV_CONST. Pre-trained models were trained on data divided by this. (recommended)
        :return tuple[np.ndarray]: returns x_train, train labels, x_test, and test labels.'''
        if isinstance(dataset, pd.DataFrame):
            if column is None:
                dataset = dataset.to_numpy(dtype=dtype).flatten()
            else:
                dataset = dataset[column].to_numpy(dtype=dtype).flatten()
        elif isinstance(dataset, pd.Series):
            if column is None:
                dataset = dataset.to_numpy(dtype=dtype).flatten()
            else:
                dataset = dataset[column].to_numpy(dtype=dtype).flatten()
        elif isinstance(dataset, np.ndarray) or (hasattr(dataset, '__iter__') and not(isinstance(dataset, str))):
            dataset = np.array(dataset, dtype=dtype, copy=True).flatten()
        elif isinstance(dataset, str):
            def try_into(func):
                try:
                    f = func(dataset)
                    if column is None:
                        return f.to_numpy(dtype=dtype).flatten()
                    return f[column].to_numpy(dtype=dtype).flatten()
                except Exception:
                    pass
            for func in {pd.read_csv, pd.read_excel, pd.read_feather, pd.read_fwf, pd.read_hdf, pd.read_html, pd.read_json, pd.read_orc, pd.read_pickle, pd.read_spss, pd.read_parquet, pd.read_stata, pd.read_xml}:
                d = try_into(func)
                if isinstance(d, np.ndarray):
                    dataset = d.copy()
                    break
            if isinstance(dataset, str):
                raise ValueError('Cannot open and convert file-path specified: %s' % dataset)
        x_train = []
        y_train = []
        for i  in range(round(len(dataset)-(size))):
            x_train += [dataset[i:i+size]]
            y_train += [dataset[i+size]]
        x_train = np.array(x_train, dtype=dtype, copy=True)
        y_train = np.array(y_train, dtype=dtype, copy=True)
        if div_const:
            x_train = x_train / cls.DIV_CONST
            y_train = y_train / cls.DIV_CONST
        x_train = x_train.reshape(x_train.shape[0], size, 1)
        x_test = x_train[0-abs(last)::]
        x_train = x_train[:0-abs(last)]
        y_test = y_train[0-abs(last)::]
        y_train = y_train[:0-abs(last)]
        return x_train, y_train, x_test, y_test
    
    
    def save_model(self, dir: str='./model'):
        '''Saves this model in the dir specified.
        :param dir: The directory to save this model to.'''
        self.model.save(dir, True, True)


    def save_weights(self, filepath: str='./model.h5'):
        '''Saves the models weights to the filepath provided.
        :param filepath: The filepath to save the weights to.
        :return None:'''
        self.model.save_weights(filepath, True)
        
    
    def load_model(self, dir: str='./model'):
        '''Loads a tensorflow.keras.Sequential model from dir provided.
        :param dir: The path to a directoy to load a tensorflow.keras.Sequential object from.
        :return tensorflow.keras.Sequential:'''
        model: Sequential = load_model(dir, compile=False) # type: ignore
        self.model = model
        self.model.compile(self.optimizer, loss=self.loss, metrics=['acc'],
                           jit_compile=True, )
        return model

    def load_weights(self, filepath: str='./model.h5'):
        '''Loads weights to the current model.
        :param filepath: A string to the filepath of the weights file to load.
        :return None:'''
        
        self.model.load_weights(filepath)
    
        
    def train(self, x_train, y_train, epochs: int=5, batch_size: int=32, *args, **kwargs) -> None:
        '''Fits the model created by reset_model() with the data and labels provided
        (self.reset_model() must be run before this.)
        :param self: An instantiated predictor object.
        :param x_train: Training data.
        :param y_train: Labels to associate with the data (not classify).
        :param epochs: The amount of epochs to train on the data.
        :param batch_size: The batch size to batch the training data on.
        :param *args: Extra args to pass to tensorflow.keras.Sequential.fit.
        :param **kwargs: Extra kwargs to pass to tensorflow.keras.Sequential.fit.
        :return tensorflow.keras.callbacks.History:'''
        
        self.model.summary()
        return self.model.fit(x_train, y_train, *args, batch_size=batch_size, epochs=epochs, use_multiprocessing=True, max_queue_size=100, workers=100, **kwargs)
        
        
    def __init__(self, size: int=24, loss: str | losses.Loss='huber', optimizer: str | optimizers.Optimizer='nadam', activation: str ='elu'):
        '''Instantiates a predictor object.
        :param size: An integer that is the base size for the model passed, it's layers, data splitting, fitting and prediction making.
        :param loss: A string or tensorflow.keras.losses.Loss object that is the loss algorithim for the model.
        :param optimizer: A string or tensorflow.keras.optimizers.Optimizer object that is the optimizer for the model.
        :param acitvation: A string that is the activation function for all layers in the model.'''
        self.size = size
        self.loss = loss
        self.optimizer = optimizer
        self.activation = activation
        self.model = Sequential()
        
    @classmethod
    def _load_pred(cls, dataset: str | pd.Series | pd.DataFrame | np.ndarray, dtype: np.dtype=np.float16, column: str | None=None, size: int=24, div_const: bool=True) -> tuple[np.ndarray]:
        '''Runs an experimental prediction algorithm on the dataset given.
        :param cls: The predictor class.
        :param dataset: An iterable, pandas Series/DataFrame, or a string to a filepath to be loaded via-pandas.
        :param dtype: A np.dtype for the dtype of the loaded dataset.
        :param column: A string. If dataset is a pandas Series/DataFrame or loaded from a filepath via-pandas, the column is selected for the data.
        :param size: The last number of items to predict from the dataset.
        :param div_const: Whether or not to divide all data in the dataset by the class DIV_CONST. Pre-trained models were trained on data divided by this. (recommended)
        :return:'''
        if isinstance(dataset, pd.DataFrame):
            if column is None:
                dataset = dataset.to_numpy(dtype=dtype).flatten()
            else:
                dataset = dataset[column].to_numpy(dtype=dtype).flatten()
        elif isinstance(dataset, pd.Series):
            if column is None:
                dataset = dataset.to_numpy(dtype=dtype).flatten()
            else:
                dataset = dataset[column].to_numpy(dtype=dtype).flatten()
        elif isinstance(dataset, np.ndarray) or (hasattr(dataset, '__iter__') and not(isinstance(dataset, str))):
            dataset = np.array(dataset, dtype=dtype, copy=True).flatten()
        elif isinstance(dataset, str):
            def try_into(func):
                try:
                    f = func(dataset)
                    if column is None:
                        return f.to_numpy(dtype=dtype).flatten()
                    return f[column].to_numpy(dtype=dtype).flatten()
                except Exception:
                    pass
            for func in {pd.read_csv, pd.read_excel, pd.read_feather, pd.read_fwf, pd.read_hdf, pd.read_html, pd.read_json, pd.read_orc, pd.read_pickle, pd.read_spss, pd.read_parquet, pd.read_stata, pd.read_xml}:
                d = try_into(func)
                if isinstance(d, np.ndarray):
                    dataset = d.copy()
                    break
            print(d.shape, type(d))
            if isinstance(dataset, str):
                raise ValueError('Cannot open and convert file-path specified: %s' % dataset)
        x_test = []
        for i  in range(round(len(dataset)-(size))):
            x_test += [dataset[i:i+size]]
        x_test = np.array(x_test, copy=True, dtype=dtype)
        if div_const:
            x_test: np.ndarray = x_test / cls.DIV_CONST
        x_test = x_test.reshape(x_test.shape[0], size, 1)
        return x_test
    
    
    def predict(self, dataset: str | pd.Series | pd.DataFrame | np.ndarray, dtype: np.dtype=np.float16, column: str | None=None, size: int=24, div_const: bool=True) -> np.ndarray:
        '''Runs a prediction on the dataset given.
        :param self: An instantiated predictor object.
        :param dataset: An iterable, pandas Series/DataFrame, or a string to a filepath to be loaded via-pandas.
        :param dtype: A np.dtype for the dtype of the loaded dataset.
        :param column: A string. If dataset is a pandas Series/DataFrame or loaded from a filepath via-pandas, the column is selected for the data.
        :param size: The last number of items to predict from the dataset.
        :param div_const: Whether or not to divide each entry of the dataset by what the test models were divided by.
        :return np.ndarray:'''
        x_test = self._load_pred(dataset, dtype, column, self.size, div_const)[-size::]
        return self.model.predict(x_test).flatten()
    
    
    def experimental_predict(self, dataset: str | pd.Series | pd.DataFrame | np.ndarray, dtype: np.dtype=np.float16, column: str | None=None, size: int=24, div_const: bool=True) -> np.ndarray:
        '''Runs an experimental prediction algorithm on the dataset given.
        :param self: An instantiated predictor object.
        :param dataset: An iterable, pandas Series/DataFrame, or a string to a filepath to be loaded via-pandas.
        :param dtype: A np.dtype for the dtype of the loaded dataset.
        :param column: A string. If dataset is a pandas Series/DataFrame or loaded from a filepath via-pandas, the column is selected for the data.
        :param size: The last number of items to predict from the dataset
        :param div_const: Whether or not to divide each entry of the dataset by what the test models were divided by.
        :return np.ndarray:'''
        x_test: np.ndarray = self._load_pred(dataset, dtype, column, self.size, div_const)[-self.size:]
        predictions: np.ndarray = x_test[-1].reshape(1, self.size, 1)
        iters = []
        for i in range(size):
            step = self.model.predict(predictions)
            predictions = (predictions + step)[-self.size::]
            iters.append(step[0])
        return np.array(iters, dtype=dtype).flatten()

    def pickle_model(self, filepath: str='./model.pickle'):
        '''Saves the current model in a pickle format.
        :param self: An instantiated predictor object.
        :param filepath: The filepath to save the pickled model data to.
        :return None:'''
        pk.dump(self.model, open(filepath, 'wb'))

    def unpickle_model(self, filepath: str='./model.pickle') -> Sequential:
        '''Loads a model that was saved in a pickle format.
        :param self: An instantiated predictor object.
        :param filepath: The filepath to load the pickled model data from.
        :return tensorflow.keras.Sequential:'''
        model: Sequential = pk.load(open(filepath, 'rb'))
        self.model = model
        return model
        
    @staticmethod
    def plot(lines: typing.Iterable, names: typing.Iterable[typing.AnyStr]=[''], title: str='plot', scatter: bool=False, xlabel: str='Days', ylabel: str='Price'):
        '''Plots the data given.
        :param lines: An iterable of data to scatter or plot as a line.
        :param names: An iterable of strings to associate names to scatters or lines.
        :param scatter: Whether or not to plot a scatter graph.
        :param xlabel: The xlabel of the shown figure.
        :param ylabel: The ylabel of the shown figure.
        :return None:
        '''
        if len(names) < len(lines):
            names = [*names] + ((len(lines) - len(names))*[''])
        lines = zip(lines, names,)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for line, name in lines:
            if scatter:
                plt.scatter(*line, label=name,)
            else:
                plt.plot(line, label=name,)
        plt.show()
        
        
# some test code!
p = predictor(180)
p.reset_model()
p.load_weights('./weights/huber_wmt_180.h5')
x_train, y, x_test, y_test = predictor._load_data('./WMT.csv', 180, 'Open', size=180)
prediction = p.predict('./WMT.csv', column='Open', size=180)
predictor.plot((prediction, y_test), ('Predicted', 'Real'))