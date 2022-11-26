# weights
## loading
To load and use pretrained weights, follow this example.
```python

pred = predictor(365)  # instantiate a predictor that splits data into 365 days
pred.reset_model()  # instantiate a tensorflow.keras.Sequential model that takes in 365 day split data
pred.load_weights('./weights/huber_wmt_365.h5')  # load up trained weights from a model that accepts 365 day split data
x_train, y_labels, x_test, y_test = predictor._load_data('./WMT.csv', last=365, column='Open', size=365)  # load up some testing data, split into 365 days and take the last 365 days for predicting.
prediction = pred.predict('./WMT.csv', column='Open', size=365)  # predict the next 365 days after the 39th year (the last year was ignored in training)
predictor.plot((prediction, y_test), ('Predicted', 'Real'))  # plot the predictions V.S the actual prices
```
## training
To train your own model on custom data, follow the example below.
```python
pred = predictor(60)  # instantiate a predictor that takes in data split by 60 day intervals
pred.reset_model()  # instantiate a tensorflow.keras.Sequential model that takes in 60 day split data
x_train, y_labels, x_test, y_test = predictor._load_data('./Tor_Ont_Can_temp.csv', last=60, column='temperature', size=60)  # load up training data, labels, and testing data and labels
# the following division is not necessary, but it keeps data clean.
DIV = 10 ** len(str((x_train + y_labels + x_test + y_test).max())) # the amount to divide the data by
x_train = x_train / DIV 
y_train = y_train / DIV
x_test = x_test / DIV 
y_test = y_test / DIV
open('./data.pickle', 'x')  # create a pickle file
pickle.dump((x_train, y_labels, x_test, y_test), open('./data.pickle', 'wb'))  # save the data
print(DIV)  # to note down the division constant
predictor.train(x_train, y_labels, div_const=False, epochs=5)  # train the model for 5 epochs (do not divide the data further)
predictor.save_model(dir='./temperature_predictor')  # save the trained model
predictions = predictor.predict(x_test, size=365)  # evaluate the data.
predictor.plot((predictions, y_test), ('Predicted', 'Real'))  # plot the data
```
Something to note, is that when training your model, the last n (n being the data split size) is ignored in the dataset. Because the -n index requires n more indices afterward to create a label.
In simple terms, the last n (n being amount the data is split by) is ignored in training.