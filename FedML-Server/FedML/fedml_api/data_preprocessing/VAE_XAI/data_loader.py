import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def data_loader(config):
    if config["data_dir"] == 'scada_water_level':
        data_dir = "../VAE-XAI-related/datasets/{}/{}.csv".format(config["data_dir"], config["dataset"])
        df = pd.read_csv(data_dir, header=0, index_col=0)
        df = df.dropna()
        train = df[0:2000]
        test = df[2000:]
        data = df.iloc[:, 0:10]
        train_data = train.iloc[:,0:10]
        test_data = test.iloc[:,0:10]
        train_labels = train.iloc[:,10]
        test_labels = test.iloc[:,10]

        train_labels = train_labels.astype(bool)
        test_labels = test_labels.astype(bool)

        normal_train_data = train_data[train_labels]
        normal_test_data = test_data[test_labels]

        # anomalous_train_data = train_data[~train_labels]
        # anomalous_test_data = test_data[~test_labels]

        return normal_train_data, normal_test_data, test_data, test_labels, data

    elif config["data_dir"] == 'SWaT':
        data_dir = "../VAE-XAI-related/datasets/{}/{}.csv".format(config["data_dir"], config["dataset"])
        df = pd.read_csv(data_dir, header=0, index_col=0)
        df = df.dropna()
        data = df.iloc[:,:-1]
        labels = df.iloc[:,-1]

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        train_labels = train_labels.astype(bool)
        test_labels = test_labels.astype(bool)

        normal_train_data = train_data[train_labels]
        normal_test_data = test_data[test_labels]

        # anomalous_train_data = train_data[~train_labels]
        # anomalous_test_data = test_data[~test_labels]

        return normal_train_data, normal_test_data, test_data, test_labels, data

    elif config["data_dir"] == 'scada_gas_pipeline':
        data_dir = "../VAE-XAI-related/datasets/{}/{}.npz".format(config["data_dir"], config["dataset"])
        data = np.load(data_dir)
        train_data = (data['training'] - data['train_m'])/data['train_std']
        test_data = (data['test'] - data['train_m'])/data['train_std']
        # Normalize data
        train_data = pd.DataFrame(train_data)
        test_set = pd.DataFrame(test_data)
        # Prepare test set labels
        test_labels= pd.DataFrame(0, index=np.arange(len(test_set)), columns=[0])
        test_labels.iloc[:] = True
        test_labels.iloc[data['idx_anomaly_test']] = False
        # Split Train/Validation set
        train_set, val_set = train_test_split(train_data, test_size=0.2, shuffle=True)
        return train_set, val_set, test_set, test_labels, data

