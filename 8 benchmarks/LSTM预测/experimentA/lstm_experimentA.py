
# coding: utf-8

# In[26]:



# -*- coding:utf-8 -*-
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.layers import GRU
from keras.layers import MaxPooling1D
from keras.layers import Dropout


dropout_rate = 0.3


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true
    #print np.abs((y_true - y_pred) / y_true)
    m = np.abs((y_true - y_pred) / y_true)
    return m


def mape(n, label, prediction):
    j = 0
    m = 0

    #j.astype('float64')
    for i in range(0, n, 1):
        #mapen = mean_absolute_percentage_error(label[i], prediction[i])
        j=j+mean_absolute_percentage_error(label[i], prediction[i])[0]
    m=float(j/n)
    return m



train_dataset = read_csv('trainingA.csv',header=0,index_col=None)
train_values = train_dataset.values


test_dataset = read_csv('testingA.csv', header=0, index_col=None)
test_values = test_dataset.values

train_X, train_y = train_values[:, 1:4], train_values[:, 0]

test_X, test_y = test_values[:, 1:4], test_values[:, 0]


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


print(train_X.shape[1], train_X.shape[2])

model = Sequential()
model.add(LSTM(16, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True,activation='relu'))
model.add(Dropout(dropout_rate))
model.add(LSTM(64,activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

optimizer = SGD(lr=0.001, momentum=0.9)
model.compile(loss='msle', optimizer=optimizer)
# fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=8, verbose=2, shuffle=False)


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 0:]), axis=1)
inv_yhat = inv_yhat[:, 0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 0:]), axis=1)
inv_y = inv_y[:, 0]

l = len(test_y)
single_mape = mean_absolute_percentage_error(test_y, yhat)
for index in range(len(single_mape)):
    single_mape[index] = round(float(single_mape[index]),2)

para1 = []
para2 = []
para3 = []
for index in range(len(test_y)):
    para1.append(test_y[index][0])

for index in range(len(yhat)):
    para2.append(yhat[index][0])

for index in range(len(single_mape)):
    para3.append(single_mape[index][0])


Data = {'reality':para1,'predict': para2, 'mape': para3}
df = DataFrame(Data, columns=["reality", "predict", "mape"])
df.to_csv("experimentA_LSTM_result.csv", index=False)

average_mape = mape(l, test_y, yhat)
print('average mape is'+"%.4f" % average_mape)


