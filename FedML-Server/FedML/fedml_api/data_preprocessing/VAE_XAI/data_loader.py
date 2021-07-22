import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def data_loader(data_dir):
    df = pd.read_csv(data_dir, header=0)
    df = df.dropna()
    # The last element contains the labels
    labels = df.iloc[:, 10]
    labels = labels.where(labels=='No', 0)
    labels = labels.where(labels==0, 1)
    # The other data points are the electrocadriogram data
    data = df.iloc[:, 0:10]
    #Chia bo train bo test
    train_data, test_data, train_labels, test_labels = train_test_split( data, labels, test_size=0.3, random_state=20)

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    #Chuẩn hóa dữ liệu
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    return normal_train_data, normal_test_data, test_data, test_labels, data
