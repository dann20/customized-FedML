from FedML.fedml_core.trainer.base_vae_lstm import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig


class DataGenerator(BaseDataGenerator):
    def __init__(self, config, num_client):
        super(DataGenerator, self).__init__(config)
        # load data here: generate 3 state variables: train_set, val_set and test_set
        self.num_client = num_client # sua dong nay
        self.load_NAB_dataset(self.config['dataset'], self.config['y_scale'])

    def load_NAB_dataset(self, dataset, y_scale=6):
        if dataset == 'ecg':
            data_dir = '../VAE-LSTM-related/datasets/NAB-known-anomaly/'
            # data_dir = '/home/pi/customized-FedML/FedML-IoT/raspberry_pi/VAE-LSTM-related/datasets/NAB-known-anomaly/'
            data = np.load(data_dir + dataset + '_{}.npz'.format(self.num_client))

            # normalise the dataset by training set mean and std
            train_m = data['train_m']
            train_std = data['train_std']
            readings_normalised = (data['readings'] - train_m) / train_std

            # slice training set into rolling windows
            n_train_sample = len(data['training'])
            print("N_TRAIN_SAMPLE: ", n_train_sample)
            dataclient = data['training']
            n_train_vae = n_train_sample - self.config['l_win'] + 1
            rolling_windows = np.zeros((n_train_vae, self.config['l_win'], self.config['n_channel'])) #them n_channel dong nay
            for i in range(n_train_sample - self.config['l_win'] + 1):
                rolling_windows[i] = np.reshape(data['training'][i:i + self.config['l_win']],(self.config['l_win'], self.config['n_channel']))

            # create VAE training and validation set
            idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
            # if self.config['n_channel']==1:
            self.train_set_vae = dict(data=np.reshape(rolling_windows[idx_train],(-1, self.config['l_win'], self.config['n_channel'])))
            self.val_set_vae = dict(data=np.reshape(rolling_windows[idx_val],(-1, self.config['l_win'], self.config['n_channel'])))
            self.test_set_vae = dict(data=np.reshape(rolling_windows[idx_val[:self.config['batch_size']]], (-1, self.config['l_win'], self.config['n_channel'])))
            # else: #them dong nay
            #     self.train_set_vae = dict(data=rolling_windows[idx_train])
            #     self.val_set_vae = dict(data=rolling_windows[idx_val])
            #     self.test_set_vae = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])
            print("shape of train,val,test set vae:",self.train_set_vae['data'].shape,\
                self.val_set_vae['data'].shape,self.test_set_vae['data'].shape)

            # create LSTM training and validation set
            for k in range(self.config['l_win']):
                n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
                n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
                print("DATA: ", dataclient)
                print("N_TRAIN_LSTM: ", n_train_lstm)
                cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win'], self.config['n_channel'])) #them n_channel dong nay
                for i in range(n_train_lstm):
                    cur_seq = np.zeros((self.config['l_seq'], self.config['l_win'],self.config['n_channel'])) #them n_channel dong nay
                    for j in range(self.config['l_seq']):
                        # print(k,i,j)
                        cur_seq[j] = np.reshape(dataclient[k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]\
                                                ,(self.config['l_win'], self.config['n_channel']))#sua tu data['training'] thanh dataclient
                    cur_lstm_seq[i] = cur_seq
                if k == 0:
                    lstm_seq = cur_lstm_seq
                else:
                    lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)
            n_train_lstm = lstm_seq.shape[0]
            idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
            # if self.config['n_channel'] == 1: #them dong nay
            #self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
            #self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
            # else:
            self.train_set_lstm = dict(data=lstm_seq[idx_train])
            self.val_set_lstm = dict(data=lstm_seq[idx_val])
            print("shape of train, val set lstm:",self.train_set_lstm['data'].shape,self.val_set_lstm['data'].shape)

        elif 'scada' in dataset: # divided dataset for each client beforehand
            data_dir = '../VAE-LSTM-related/datasets/NAB-known-anomaly/'
            data = np.load(data_dir + dataset + '.npz')

            # slice training set into rolling windows
            n_train_sample = len(data['training'])
            # n_train_sample = int(n_train_sample*0.25)
            # dataclient = data['training'][( n_train_sample*(self.num_client-1) ):( n_train_sample*self.num_client )]
            stride_ori = data['training'].reshape((-1,self.config['n_channel'])).strides
            strides = np.insert(stride_ori, 0, stride_ori[0], axis = 0)
            n_train_vae = n_train_sample - self.config['l_win'] + 1
            shape = [n_train_vae, self.config['l_win'], self.config['n_channel']]
            rolling_windows = np.lib.stride_tricks.as_strided(data['training'],shape, strides, writeable=False)

            # create VAE training and validation set
            idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
            self.train_set_vae = dict(data=np.reshape(rolling_windows[idx_train],(-1, self.config['l_win'], self.config['n_channel'])))
            self.val_set_vae = dict(data=np.reshape(rolling_windows[idx_val],(-1, self.config['l_win'], self.config['n_channel'])))
            self.test_set_vae = dict(data=np.reshape(rolling_windows[idx_val[:self.config['batch_size']]], (-1, self.config['l_win'], self.config['n_channel'])))
            print("shape of train,val,test set vae:",self.train_set_vae['data'].shape,\
                self.val_set_vae['data'].shape,self.test_set_vae['data'].shape)

            # create LSTM training and validation set
            for k in range(self.config['l_win']):
              n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
              n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
              shape = [n_train_lstm, self.config['l_seq'], self.config['l_win'], self.config['n_channel']]
              strides = np.insert(stride_ori, 0, [stride_ori[0]*self.config['l_win'], stride_ori[0]*self.config['l_win']], axis = 0)
              cur_lstm_seq = np.lib.stride_tricks.as_strided(data['training'][k:],shape, strides, writeable = False)
              if k == 0:
                lstm_seq = cur_lstm_seq
              else:
                lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)

            n_train_lstm = lstm_seq.shape[0]
            idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
            self.train_set_lstm = dict(data=lstm_seq[idx_train])
            self.val_set_lstm = dict(data=lstm_seq[idx_val])
            print("shape of train, val set lstm:",self.train_set_lstm['data'].shape,self.val_set_lstm['data'].shape)

        else:
            data_dir = '../VAE-LSTM-related/datasets/NAB-known-anomaly/'
            # data_dir = '/home/pi/customized-FedML/FedML-IoT/raspberry_pi/VAE-LSTM-related/datasets/NAB-known-anomaly/'
            data = np.load(data_dir + dataset + '.npz')

            # normalise the dataset by training set mean and std
            train_m = data['train_m']
            train_std = data['train_std']
            readings_normalised = (data['readings'] - train_m) / train_std

            # plot normalised data
            # fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
            # fig.subplots_adjust(hspace=.4, wspace=.4)
            # axs.plot(data['t'], readings_normalised)
            # if data['idx_split'][0] == 0:
            #     axs.plot(data['idx_split'][1] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
            # else:
            #     for i in range(2):
            #         axs.plot(data['idx_split'][i] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
            # axs.plot(*np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')
            # for j in range(len(data['idx_anomaly'])):
            #     axs.plot(data['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')
            # axs.grid(True)
            # axs.set_xlim(0, len(data['t']))
            # axs.set_ylim(-y_scale, y_scale)
            # axs.set_xlabel("timestamp (every {})".format(data['t_unit']))
            # axs.set_ylabel("readings")
            # #axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset, train_m, train_std))
            # axs.legend(('data', 'train test set split', 'anomalies'))
            # savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')

            # slice training set into rolling windows
            n_train_sample = len(data['training'])
            # them doan nay
            n_train_sample = int(n_train_sample*0.3)
            if self.num_client==1:
                dataclient = data['training'][:n_train_sample]
            else:
                dataclient = data['training'][n_train_sample:]
                n_train_sample = len(data['training']) - n_train_sample
            n_train_vae = n_train_sample - self.config['l_win'] + 1
            rolling_windows = np.zeros((n_train_vae, self.config['l_win'], self.config['n_channel'])) #them n_channel dong nay
            for i in range(n_train_sample - self.config['l_win'] + 1):
                rolling_windows[i] = np.reshape(data['training'][i:i + self.config['l_win']],(self.config['l_win'], self.config['n_channel']))

            # create VAE training and validation set
            idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
            # if self.config['n_channel']==1:
            self.train_set_vae = dict(data=np.reshape(rolling_windows[idx_train],(-1, self.config['l_win'], self.config['n_channel'])))
            self.val_set_vae = dict(data=np.reshape(rolling_windows[idx_val],(-1, self.config['l_win'], self.config['n_channel'])))
            self.test_set_vae = dict(data=np.reshape(rolling_windows[idx_val[:self.config['batch_size']]], (-1, self.config['l_win'], self.config['n_channel'])))
            # else: #them dong nay
            #     self.train_set_vae = dict(data=rolling_windows[idx_train])
            #     self.val_set_vae = dict(data=rolling_windows[idx_val])
            #     self.test_set_vae = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])
            print("shape of train,val,test set vae:",self.train_set_vae['data'].shape,\
                self.val_set_vae['data'].shape,self.test_set_vae['data'].shape)

            # create LSTM training and validation set
            for k in range(self.config['l_win']):
                n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
                n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
                cur_lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win'], self.config['n_channel'])) #them n_channel dong nay
                for i in range(n_train_lstm):
                    cur_seq = np.zeros((self.config['l_seq'], self.config['l_win'],self.config['n_channel'])) #them n_channel dong nay
                    for j in range(self.config['l_seq']):
                        # print(k,i,j)
                        cur_seq[j] = np.reshape(dataclient[k + self.config['l_win'] * (j + i): k + self.config['l_win'] * (j + i + 1)]\
                                                ,(self.config['l_win'], self.config['n_channel']))#sua tu data['training'] thanh dataclient
                    cur_lstm_seq[i] = cur_seq
                if k == 0:
                    lstm_seq = cur_lstm_seq
                else:
                    lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)
            n_train_lstm = lstm_seq.shape[0]
            idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
            # if self.config['n_channel'] == 1: #them dong nay
            #self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
            #self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
            # else:
            self.train_set_lstm = dict(data=lstm_seq[idx_train])
            self.val_set_lstm = dict(data=lstm_seq[idx_val])
            print("shape of train, val set lstm:",self.train_set_lstm['data'].shape,self.val_set_lstm['data'].shape)

    def plot_time_series(self, data, time, data_list):
        fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
        fig.subplots_adjust(hspace=.8, wspace=.4)
        axs = axs.ravel()
        for i in range(4):
            axs[i].plot(time / 60., data[:, i])
            axs[i].set_title(data_list[i])
            axs[i].set_xlabel('time (h)')
            axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
        savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')
