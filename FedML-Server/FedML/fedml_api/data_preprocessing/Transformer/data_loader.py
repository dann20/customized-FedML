import logging

import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config, mode='train', model_type="autoencoder"):
        super().__init__()
        self.config = config
        self.mode = mode
        self.model_type = model_type
        if model_type == "autoencoder":
            self.load_dataset(self.config["auto_dataset"])
        else:
            self.load_dataset(self.config["trans_dataset"])

    def __len__(self):
        return self.rolling_windows.shape[0]

    def __getitem__(self, index):
        if (self.mode != 'test') or (self.model_type == "autoencoder"):
            inp = target = self.rolling_windows[index, :, :]
        else:
            inp = self.rolling_windows[index, :, :]
            target = self.rolling_windows[index,
                                          self.config["pre_mask"]:self.config["post_mask"], :]
        sample = {"input": inp, "target": target}
        return sample

    def load_dataset(self, dataset):
        data_dir = "../Transformer-related/datasets/{}/".format(self.config["data_dir"])
        self.data = np.load(data_dir + dataset + ".npz")

        if self.mode == 'train':
            if len(self.data['training'].shape) == 1:
                data = np.expand_dims(self.data['training'], -1)
            else:
                data = self.data['training']
            if int(data.shape[0] * 0.1) > self.config['autoencoder_dims']:
                data = data[:int(data.shape[0] * 0.9), :]
            logging.info("TRAINING DATA SHAPE: {}".format(data.shape))
        elif self.mode == 'validate':
            if len(self.data['training'].shape) == 1:
                data = np.expand_dims(self.data['training'], -1)
            else:
                data = self.data['training']
            if int(data.shape[0] * 0.1) > self.config['autoencoder_dims']:
                data = data[int(data.shape[0] * 0.9):, :]
            else:
                self.rolling_windows = np.zeros((0,0,0))
                return
            logging.info("VALIDATION DATA SHAPE: {}".format(data.shape))
        else:
            if len(self.data['test'].shape) == 1:
                data = np.expand_dims(self.data['test'], -1)
            else:
                data = self.data['test']
            logging.info("TEST DATA SHAPE: {}".format(data.shape))

        # slice training set into rolling windows
        if self.model_type == "autoencoder":
            self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                data, self.config["autoencoder_dims"], axis=0, writeable=True
            ).transpose(0, 2, 1)
        else:
            self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                data, self.config["l_win"], axis=0, writeable=True
            ).transpose(0, 2, 1)
