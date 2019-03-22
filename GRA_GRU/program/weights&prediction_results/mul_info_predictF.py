import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from numpy import *
import numpy as np
import mul_info_weightF

def mul_info_predict(gru_test_file,le_test_file):
    gru_test_dataset = read_csv(gru_test_file, header=0, index_col=None)
    reality = gru_test_dataset.values[:,0]
    gru_predict = gru_test_dataset.values[:, 1]

    le_test_dataset = read_csv(le_test_file, header=0, index_col=None)
    le_predict = le_test_dataset.values[:, 1]

    mul_info_predict = np.array(gru_predict) * mul_info_weightF.gru_weight + np.array(le_predict) * mul_info_weightF.le_weight
    mape_list = []
    acc_list  = []
    mape = 0
    for i in range(len(mul_info_predict)):
        mape_list.append(abs(mul_info_predict[i]-reality[i])/reality[i])
        acc_list.append(1-abs(mul_info_predict[i]-reality[i])/reality[i])
        mape += abs(mul_info_predict[i]-reality[i])/reality[i]
    mape = mape/len(mul_info_predict)
    print(mape)
    print(np.var(acc_list))

if __name__ == "__main__":
    mul_info_predict("experimentF_GRU_result.csv","experimentF_LE_result.csv")