import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def load_data_LCHA(train_data_dir, test_data_dir):
    # train files and test files are separate
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.xlsx')]
    train_data = pd.DataFrame()
    train_label = pd.DataFrame()
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        file_data = pd.read_excel(file_path, header = None, engine='openpyxl')
        X = file_data.iloc[:,:-1]
        y = file_data.iloc[:,-1]
        # train_data = train_data.append(X)
        # train_label = train_label.append(y)
        train_data = pd.concat([train_data, X])
        train_label = pd.concat([train_label, y])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.xlsx')]
    test_data = pd.DataFrame()
    test_label = pd.DataFrame()
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        file_data = pd.read_excel(file_path, header = None, engine='openpyxl')
        X = file_data.iloc[:,:-1]
        y = file_data.iloc[:,-1]
        # test_data = test_data.append(X)
        # test_label = test_label.append(y)
        test_data = pd.concat([test_data, X])
        test_label = pd.concat([test_label, y])
    class_num = 11

    return train_data, train_label, test_data, test_label, class_num

