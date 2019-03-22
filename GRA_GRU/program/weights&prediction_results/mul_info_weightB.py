import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from numpy import *
import numpy as np


def mul_info_weightB(gru_valid_file, le_valid_file):
    gru_valid_dataset = read_csv(gru_valid_file, header=0, index_col=None)
    gru_accurary = gru_valid_dataset.values[:, 3]

    le_valid_dataset = read_csv(le_valid_file, header=0, index_col=None)
    le_accurary = le_valid_dataset.values[:, 3]

    #得出gru准确率的频数
    gru_dic = {}
    for i,a in enumerate(gru_accurary):
        if a not in gru_dic:
            gru_dic[a] = 1
        else:
            gru_dic[a] += 1
    gru_accurary_frequency = []
    for i,a in enumerate(gru_accurary):
        gru_accurary_frequency.append(gru_dic[a])

    #gru频数所占概率
    gru_frequency_probability = np.array(gru_accurary_frequency)/len(gru_accurary_frequency)

    #gru概率的对数
    gru_probability_log = log2(gru_frequency_probability)

    #gru的自信息熵
    gru_self_entrocy = - (gru_frequency_probability * gru_probability_log)





    # 得出le准确率的频数
    le_dic = {}
    for i, a in enumerate(le_accurary):
        if a not in le_dic:
            le_dic[a] = 1
        else:
            le_dic[a] += 1
    le_accurary_frequency = []
    for i, a in enumerate(le_accurary):
        le_accurary_frequency.append(le_dic[a])

    # le频数所占概率
    le_frequency_probability = np.array(le_accurary_frequency) / len(le_accurary_frequency)

    # le概率的对数
    le_probability_log = log2(le_frequency_probability)

    # le的自信息熵
    le_self_entrocy = - (le_frequency_probability * le_probability_log)

    #gru与le准确率的乘积
    mul_accurary_multiply = gru_accurary * le_accurary

    #gru与le准确率的商
    mul_accurary_division = gru_accurary/le_accurary

    #乘积与商的和
    multiply_add_division = mul_accurary_multiply + mul_accurary_division

    #公共频数
    mul_dic = {}
    for i, a in enumerate(multiply_add_division):
        if a not in mul_dic:
            mul_dic[a] = 1
        else:
            mul_dic[a] += 1
    mul_frequency = []
    for i, a in enumerate(multiply_add_division):
        mul_frequency.append(mul_dic[a])

    #共概率
    mul_probability = np.array(mul_frequency)/len(mul_frequency)

    #共概率除以单概率
    mul_division_single = mul_probability/(gru_frequency_probability * le_accurary_frequency)

    #对mul_division_single取对数
    mul_division_single_log = log2(mul_division_single)

    #互信息
    mutual_information = -mul_probability * mul_division_single_log

    #根据gru_self_entrocy, le_self_entrocy, mutual_information求权重
    gru_ratio = sum(mutual_information) / sum(gru_self_entrocy)
    le_ratio = sum(mutual_information) / sum(le_self_entrocy)

    gru_weight = gru_ratio / (gru_ratio + le_ratio)
    le_weight = le_ratio / (gru_ratio + le_ratio)

    Data = {"le_accurary":le_accurary, "le_accurary_frequency":le_accurary_frequency,"le_frequency_probability":le_frequency_probability,
            "le_probability_log":le_probability_log,"le_self_entrocy":le_self_entrocy,"gru_accurary":gru_accurary,
            "gru_accurary_frequency":gru_accurary_frequency,"gru_frequency_probability":gru_frequency_probability,
            "gru_probability_log":gru_probability_log,"gru_self_entrocy":gru_self_entrocy,
            "mul_accurary_multiply":mul_accurary_multiply,"mul_accurary_division":mul_accurary_division,
            "multiply_add_division":multiply_add_division,"mul_frequency":mul_frequency,"mul_probability":mul_probability,
            "mul_division_single":mul_division_single,"mul_division_single_log":mul_division_single_log,"mutual_information":mutual_information}
    df = pd.DataFrame(Data,columns=["le_accurary","le_accurary_frequency","le_frequency_probability","le_probability_log",
                                    "le_self_entrocy","gru_accurary","gru_accurary_frequency","gru_frequency_probability",
                                    "gru_probability_log","gru_self_entrocy","mul_accurary_multiply","mul_accurary_division",
                                    "multiply_add_division","mul_frequency","mul_probability","mul_division_single",
                                    "mul_division_single_log","mutual_information"])
    df.to_csv("experimentB_mutual_information.csv", index=False)

    print(gru_weight, le_weight)
    return gru_weight, le_weight


gru_weight, le_weight = mul_info_weightB("experimentB_GRU_validation_result.csv", "experimentB_LE_validation_result.csv")


