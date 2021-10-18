import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def data_loader(config):
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

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    return normal_train_data, normal_test_data, test_data, test_labels, data
