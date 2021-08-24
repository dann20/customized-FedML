import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, config, train=True, model_type="autoencoder"):
        super().__init__()
        self.config = config
        self.train = train
        self.model_type = model_type
        if model_type == "autoencoder":
            self.load_dataset(self.config["auto_dataset"])
        else:
            self.load_dataset(self.config["trans_dataset"])

    def __len__(self):
        return self.rolling_windows.shape[0]

    def __getitem__(self, index):
        if (self.train) or (self.model_type == "autoencoder"):
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

        # slice training set into rolling windows
        if self.model_type == "autoencoder":
            if self.train:
                self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                    self.data["training"], self.config["autoencoder_dims"], axis=0, writeable=True
                ).transpose(0, 2, 1)
            else:
                self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                    self.data["test"], self.config["autoencoder_dims"], axis=0, writeable=True
                ).transpose(0, 2, 1)
        else:
            if self.train:
                self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                    self.data["training"], self.config["l_win"], axis=0, writeable=True
                ).transpose(0, 2, 1)
            else:
                self.rolling_windows = np.lib.stride_tricks.sliding_window_view(
                    self.data["test"], self.config["l_win"], axis=0, writeable=True
                ).transpose(0, 2, 1)
