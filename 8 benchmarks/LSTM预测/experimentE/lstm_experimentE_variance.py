from numpy import *
import numpy as np
from pandas import read_csv
from pandas import DataFrame

data = read_csv('experimentE_LSTM_result.csv', header=0, index_col=None)
data = data.values[:, 2]
acc_lstm = []
for index in range(data.shape[0]):
    acc_lstm.append(1-data[index])
print(np.var(acc_lstm))