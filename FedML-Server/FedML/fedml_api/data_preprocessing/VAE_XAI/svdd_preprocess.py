import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_data(data):
    data.iloc[:, -1] = data.iloc[:, -1].astype('uint8')
    data.iloc[:,-1] = np.where(data.iloc[:,-1] == 0, -1, data.iloc[:,-1])
    normalData = data[data.iloc[:,-1] == 1]
    abnormalData = data[data.iloc[:,-1] == -1]

    trainSet, normalTest = train_test_split(normalData, test_size=0.2, shuffle=False)
    testSet = pd.concat([abnormalData, normalTest])

    trainLabel = trainSet.iloc[:, -1].to_numpy()
    trainLabel = trainLabel.reshape(-1,1)
    trainData = trainSet.iloc[:, :-1].to_numpy()
    testLabel = testSet.iloc[:, -1].to_numpy()
    testLabel = testLabel.reshape(-1,1)
    testData = testSet.iloc[:, :-1].to_numpy()

    return trainData, trainLabel, testData, testLabel
