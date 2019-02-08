# -*- coding:utf-8 -*-
import os
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from math import *
import math
from sympy import *
from sympy.abc import x,y,z,a,b,c,d
import numpy as np
import numpy
import csv
from math import e


def preprocessing(before_filename, after_filename):
    train_dataset = read_csv(before_filename, header=0, index_col=None)
    train_values = train_dataset.values

    power = train_values[:, 0]
    temperature = train_values[:, 1]
    humidity = train_values[:, 2]
    speed = train_values[:, 3]
    common = [1 for i in range(power.shape[0])]

    transfer_power = []
    transfer_temperature = []
    transfer_humidity = []
    transfer_speed = []


    except_value = -5.0

    for index in range(len(power)):
        if round(log(power[index], e), 2) == -inf:
            transfer_power.append(except_value)
        else:
            transfer_power.append(round(log(power[index], e), 2))


    for index in range(len(temperature)):
        if round(log(temperature[index], e), 2) == -inf:
            transfer_temperature.append(except_value)
        else:
            transfer_temperature.append(round(log(temperature[index], e), 2))

    for index in range(len(humidity)):
        if round(log(humidity[index], e), 2) == -inf:
            transfer_humidity.append(except_value)
        else:
            transfer_humidity.append(round(log(humidity[index], e), 2))

    for index in range(len(speed)):
        if round(log(speed[index], e), 2) == -inf:
            transfer_speed.append(except_value)
        else:
            transfer_speed.append(round(log(speed[index], e), 2))

    transfer_common = numpy.array(common)

    Data = {'power': transfer_power, 'temperature': transfer_temperature, 'humidity': transfer_humidity,
            'speed': transfer_speed, 'common': transfer_common}
    df = DataFrame(Data, columns=['power', 'temperature', 'humidity', 'speed', 'common'])
    df.to_csv(after_filename, index=False)

def train_start(train_filename):
    dt = read_csv(train_filename, header=0, index_col=False)
    df = pd.DataFrame(dt)
    i = 0
    while True:
        try:
            temp1 = df.sample(4, frac=None, replace=True, weights=None, random_state=None, axis=None)
            a = np.array(temp1.values[:, 1:])
            b = np.array(temp1.values[0:, 0])
            x = np.linalg.solve(a, b)
            dataframe = pd.DataFrame({'X1': x[0], 'X2': x[1], 'X3': x[2], 'X4': x[3]}, index=[i])
            dataframe.to_csv("Xee.csv", mode='a', header=None, index=True, sep=',')
            i = i + 1
            print('第%d组系数' % (i))
            print(x)
            j = 0
            p = numpy.array([[0], [0], [0], [0]])
            # print('下面是50组误差率')
            while True:
                try:
                    temp2 = df.sample(4, frac=None, replace=True, weights=None, random_state=None, axis=None)
                    a1 = temp2.values[:, 1:]
                    b1 = temp2.values[0:, 0]
                    m = 2.7182818

                    sj = np.dot(a1, x.reshape(4, 1))
                    # print(pow(m,sj))
                    # print(pow(2,sj))
                    # error = abs(pow(m,sj) - pow(m, b1.reshape(4, 1))) / pow(m, b1.reshape(4, 1))
                    error = abs(sj - b1.reshape(4, 1)) / b1.reshape(4, 1)

                    p = p + error
                    j = j + 1
                # print('--%d--'%(j))
                # print(error)
                except Exception as e:
                    print("出现异常的位置：\n", j + 1)
                    continue
                finally:
                    if j > 2999:
                        xx = [0]
                        for xl in p / 3000:
                            xx = xx + xl
                        t = xx / 4
                        dataframe = pd.DataFrame({'X5': t[0]}, index=[i - 1])
                        # lc.sort(["loan_amnt"], ascending=True).head(10)
                        dataframe.to_csv("Xe1.csv", mode='a', header=None, index=True, sep=',')
                        break

        except Exception as e:
            # print("出现异常的位置：\n", i+1)
            # print("Error: 此组方程无解", e)
            continue
        finally:
            if i > 199:
                break
    # with open('Xe.csv','ab',newline='',) as f:
    #     f.write(open('Xer1.csv','rb',row).read()

    lc = pd.DataFrame(pd.read_csv('Xe1.csv', header=None, names=['xulie', 'wucha']))


def valid_start(valid_filename, X):
    dt = read_csv(valid_filename, header=0, index_col=False)
    df = pd.DataFrame(dt)
    i = 0
    # p = numpy.array([0, 0, 0, 0])
    # p = p.reshape(4, 1)
    p = numpy.array([[0], [0], [0], [0]])
    while True:
        try:
            temp = df.sample(4, frac=None, replace=True, weights=None, random_state=None, axis=None)
            a = temp.values[:, 1:]
            b = temp.values[0:, 0].reshape(4, 1)

            x = np.array(X).reshape(4, 1)
            m = 2.7182818
            sj = np.dot(a, x)
            for index in range(sj.shape[0]):
                sj[index][0] = pow(m, sj[index][0])
            # print(sj)
            for index in range(b.shape[0]):
                b[index][0] = pow(m, b[index][0])
            # print(pow(m,sj))
            error = abs(sj - b) / b
            # error = abs(pow(m, sj) - pow(m, b.reshape(4, 1))) / pow(m, b.reshape(4, 1))
            p = p + error
            i = i + 1

            # print(p)

        except Exception as e:
            print("出现异常的位置：\n", i + 1)
            print("Error: 此组方程无解", e)
            continue
        finally:
            if i > 2999:
                xx = [0]
                for xl in p / 3000:
                    xx = xx + xl
                t = xx / 4
                return t
                break


def test_start(test_filename, n):
    dt = read_csv(test_filename, header=0, index_col=False)
    df = pd.DataFrame(dt)
    i = 0
    #p = numpy.array([0, 0, 0, 0])
    #p = p.reshape(4, 1)
    p = numpy.array([[0], [0], [0], [0]])
    #X = topn_para_average(n)
    #X = topn_optimal(n)
    X = para_median()
    while True:
        try:
            temp = df.sample(4, frac=None, replace=True, weights=None, random_state=None, axis=None)
            a = temp.values[:, 1:]
            b = temp.values[0:, 0].reshape(4,1)

            x = np.array(X).reshape(4, 1)
            m = 2.7182818
            sj = np.dot(a, x)
            for index in range(sj.shape[0]):
                sj[index][0] = pow(m,sj[index][0])
            #print(sj)
            for index in range(b.shape[0]):
                b[index][0] = pow(m,b[index][0])
            # print(pow(m,sj))
            error = abs(sj - b)/b
            #error = abs(pow(m, sj) - pow(m, b.reshape(4, 1))) / pow(m, b.reshape(4, 1))
            p = p + error
            i = i + 1

            #print(p)

        except Exception as e:
            print("出现异常的位置：\n", i + 1)
            print("Error: 此组方程无解", e)
            continue
        finally:
            if i > 2999:
                xx = [0]
                for xl in p / 3000:
                    xx = xx + xl
                t = xx / 4
                print(t)
                break


def topn_para_average(n):
    lc = pd.DataFrame(pd.read_csv('Xe1.csv', header=None, names=['xulie', 'wucha']))
    tdd = lc.sort_values(by='wucha', ascending=True).head(n)
    index = list(tdd['xulie'])
    dt = pd.read_csv('Xee.csv', header=None, names=['xulie', '参数1', '参数2', '参数3', '参数4'])
    df = pd.DataFrame(dt)
    para1 = 0
    para2 = 0
    para3 = 0
    para4 = 0
    for i in index:
        para1 = para1 + df.values[i][1]
        para2 = para2 + df.values[i][2]
        para3 = para3 + df.values[i][3]
        para4 = para4 + df.values[i][4]
    para1 = para1 / len(index)
    para2 = para2 / len(index)
    para3 = para3 / len(index)
    para4 = para4 / len(index)
    return [para1,para2,para3,para4]


def topn_optimal(n):
    lc = pd.DataFrame(pd.read_csv('Xe1.csv', header=None, names=['xulie', 'wucha']))
    tdd = lc.sort_values(by='wucha', ascending=True).head(n)
    index = list(tdd['xulie'])
    dt = pd.read_csv('Xee.csv', header=None, names=['xulie', '参数1', '参数2', '参数3', '参数4'])
    df = pd.DataFrame(dt)
    max = 1
    for i in index:
        X = [df.values[i][1],df.values[i][2],df.values[i][3],df.values[i][4]]
        optimal = valid_start("valid3.csv", X)
        if optimal < max:
            Y = X
            max = optimal

    return Y


def para_median():
    dt = pd.read_csv('Xee.csv', header=None, names=['xulie', '参数1', '参数2', '参数3', '参数4'])
    df = pd.DataFrame(dt)
    para1 = np.median(df.values[:, 1])
    para2 = np.median(df.values[:, 2])
    para3 = np.median(df.values[:, 3])
    para4 = np.median(df.values[:, 4])
    return [para1, para2, para3, para4]


if __name__ == "__main__":
    preprocessing("trainingF.csv", "train3.csv")   
    preprocessing("testingF.csv", "test3.csv")
    train_start("train3.csv")
    test_start("test3.csv", 20)



