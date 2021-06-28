# this script is based on notebook NAB_anomaly_detection_OCSVM_2.ipynb
# testing script for VAE-LSTM model on Raspberry Pi 4 (FL)

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure
import random
import time
import os
import sys
import subprocess
import math
from scipy.stats import norm

import tensorflow as tf
from FedML.fedml_api.data_preprocessing.VAE_LSTM.data_loader import DataGenerator
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Models import VAEmodel, lstmKerasModel
from FedML.fedml_api.distributed.fedavg.VAE_Trainer import vaeTrainer

from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import process_config, create_dirs, get_args, save_config

tf.compat.v1.disable_eager_execution()
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# slice into rolling windows and rolling sequences
def slice_rolling_windows_and_sequences(config, time_seq):
    n_sample = len(time_seq)
    time_seq = np.reshape(time_seq,(-1,config['n_channel']))
    print("The given sequence has {} samples".format(n_sample))
    n_vae_win = n_sample - config['l_win'] + 1
    shape = [n_vae_win, config['l_win'], config['n_channel']]
    stride_ori = time_seq.strides
    strides = np.insert(stride_ori, 0, stride_ori[0], axis = 0)
    #rolling_windows = np.zeros((n_vae_win, config['l_win'], config['n_channel']))
    rolling_windows = np.lib.stride_tricks.as_strided(time_seq,shape, strides, writeable = False)
    #for i in range(n_vae_win):
    #    rolling_windows[i] = time_seq[i:i + config['l_win']]
    sample_m = np.mean(rolling_windows, axis=1)
    sample_std = np.std(rolling_windows, axis=1)
    n_lstm_seq = n_sample - config['l_seq']*config['l_win']+1
    shape = [n_lstm_seq, config['l_seq'], config['l_win'], config['n_channel']]
    strides = np.insert(stride_ori, 0, [stride_ori[-1], stride_ori[0]*config['l_win']], axis = 0)
    lstm_seq = np.lib.stride_tricks.as_strided(time_seq,shape, strides, writeable = False)
    #lstm_seq = np.zeros((n_lstm_seq, config['l_seq'], config['l_win'], config['n_channel']))
    #for i in range(n_lstm_seq):
    #    cur_seq = time_seq[i:i+config['l_seq']*config['l_win']]
    #    for j in range(config['l_seq']):
    #        lstm_seq[i,j] = cur_seq[config['l_win']*j:config['l_win']*(j+1)]
    return rolling_windows, lstm_seq, sample_m, sample_std

# Evaluate ELBO and LSTM prediction error on the validation set
# evaluate some anomaly detection metrics
def evaluate_vae_anomaly_metrics_for_a_window(test_win):
    feed_dict = {model_vae.original_signal: np.expand_dims(test_win, 0),
                 model_vae.is_code_input: False,
                 model_vae.code_input: np.zeros((1, config['code_size']), dtype=np.float32)}

    # VAE reconstruction error
    recons_win_vae = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict))
    test_vae_recons_error = np.sum(np.square(recons_win_vae - test_win))

    # VAE latent embedding likelihood
    vae_code_mean, vae_code_std = sess.run([model_vae.code_mean, model_vae.code_std_dev], feed_dict=feed_dict)
    test_vae_kl = 0.5 * (np.sum(np.square(vae_code_mean)) + \
                            np.sum(np.square(vae_code_std)) - \
                            np.sum(np.log(np.square(vae_code_std))) - config['code_size'])

    # VAE ELBO loss
    sigma2 = 0.0005
    input_dims = model_vae.input_dims
    sigma_regularisor = input_dims/2. * np.log(sigma2) + input_dims * np.pi
    test_vae_elbo = test_vae_recons_error/sigma2 + test_vae_kl + sigma_regularisor
    return test_vae_recons_error, test_vae_kl, test_vae_elbo

def evaluate_lstm_anomaly_metric_for_a_seq(test_seq):
    feed_dict = {model_vae.original_signal: test_seq,
                 model_vae.is_code_input: False,
                 model_vae.code_input: np.zeros((1, config['code_size']), dtype=np.float32)}
    vae_embedding = np.squeeze(sess.run(model_vae.code_mean, feed_dict=feed_dict))
    #print(vae_embedding.shape)
    lstm_embedding = np.squeeze(lstm_nn_model.predict(np.expand_dims(vae_embedding[:config['l_seq']-1], 0), batch_size=1))
    lstm_embedding_error = np.sum(np.square(vae_embedding[1:] - lstm_embedding))
    # error_original = vae_embedding[1:] - lstm_embedding
    #print(error_original.shape)

    # LSTM prediction error
    feed_dict_lstm = {model_vae.original_signal: np.zeros((config['l_seq'] - 1, config['l_win'], config['n_channel']), dtype=np.float32),
                      model_vae.is_code_input: True,
                      model_vae.code_input: lstm_embedding}
    recons_win_lstm = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_lstm))
    lstm_recons_error = np.sum(np.square(recons_win_lstm - np.squeeze(test_seq[1:])))
    error_original = np.abs(recons_win_lstm - np.squeeze(test_seq[1:])).reshape((config['l_seq']-1,-1)) #them dong nay de tinh
    return lstm_recons_error, lstm_embedding_error, error_original

def return_anomaly_idx_by_threshold(test_anomaly_metric, threshold):
    # test_list = np.squeeze(np.ndarray.flatten(test_anomaly_metric))
    idx_error = np.squeeze(np.argwhere(test_anomaly_metric > threshold))

    if len(idx_error.shape) == 0:
        idx_error = np.expand_dims(idx_error, 0)

    return list(idx_error)

def augment_detected_idx(idx_detected_anomaly, anomaly_index):
    n_anomaly = len(anomaly_index)
    idx_detected_anomaly_extended = list(idx_detected_anomaly)
    for i in range(n_anomaly):
        #print(idx_detected_anomaly)
        for j in idx_detected_anomaly:
            if j in anomaly_index[i]:
                in_original_detection = set(idx_detected_anomaly_extended)
                currect_anomaly_win = set(anomaly_index[i])
                idx_detected_anomaly_extended = idx_detected_anomaly_extended + list(currect_anomaly_win - in_original_detection)
                #print(j)
                break

    return list(np.sort(idx_detected_anomaly_extended))


start = time.time()

# load VAE model
config = process_config('../VAE-LSTM-related/configs/scada1_config.json')
# create the experiments dirs
create_dirs([config['result_dir'], config['checkpoint_dir']])
# create tensorflow session
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# create a CNN model
model_vae = VAEmodel(config, "Global")
# create a CNN model
model_vae.load(sess)

# load LSTM model
lstm_model = lstmKerasModel("Global", config)
lstm_nn_model = lstm_model.create_lstm_model()
lstm_nn_model.summary()   # Display the model's architecture

# checkpoint path
checkpoint_path = config['checkpoint_dir_lstm'] + "cp_Global.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# load weights if possible
lstm_model.load_model(lstm_nn_model, checkpoint_path)

# load normalised time series
save_dir = '../VAE-LSTM-related/datasets/NAB-known-anomaly/'
# config['dataset'] = 'ecg_3'
dataset = config['dataset']
filename = '{}.npz'.format(dataset)
result = dict(np.load(save_dir+filename, allow_pickle=True))
result['test'] = result['test'].astype(np.float32)
test_data = np.split(result['test'], [10000, 20000, 30000, 40000, 50000, 60000])

if dataset == 'machine_temp':
    result['test'] = result['test'][0]
    result['idx_anomaly_test'] = result['idx_anomaly_test'][0]
    result['t_test'] = result['t_test'][0]

def main_func(test_set):
    test_windows, test_seq, test_sample_m, test_sample_std = slice_rolling_windows_and_sequences(config, test_set)
    test_windows = np.reshape(test_windows, (-1,config['l_win'],config['n_channel']))
    test_seq = np.reshape(test_seq, (-1,config['l_seq'],config['l_win'],config['n_channel']))
    print('test_windows ',test_windows.shape)
    print('test_seq ',test_seq.shape)

    # Evaluate the anomaly metrics on the test windows and sequences
    n_test_lstm = test_seq.shape[0]

    test_lstm_recons_error, test_lstm_embedding_error = np.zeros(n_test_lstm, dtype=np.float32), np.zeros(n_test_lstm, dtype=np.float32)
    test_lstm_error_original = np.zeros((n_test_lstm,config['l_seq']-1,config['l_win']*config['n_channel']), dtype=np.float32)
    for i in range(n_test_lstm):
        test_lstm_recons_error[i], test_lstm_embedding_error[i], test_lstm_error_original[i] = evaluate_lstm_anomaly_metric_for_a_seq(test_seq[i])
    print("All windows' reconstruction error is computed.")
    print("The total number of windows is {}".format(len(test_lstm_recons_error)))

    idx_anomaly_test = result['idx_anomaly_test']
    anomaly_index_lstm = []
    test_labels_lstm = np.zeros(n_test_lstm, dtype=np.float32)
    for i in range(len(idx_anomaly_test)):
        idx_start = idx_anomaly_test[i]-(config['l_win']*config['l_seq']-1)
        idx_end = idx_anomaly_test[i]+1
        if idx_start < 0:
            idx_start = 0
        if idx_end > n_test_lstm:
            idx_end = n_test_lstm
        anomaly_index_lstm.append(np.arange(idx_start,idx_end))
        test_labels_lstm[idx_start:idx_end] = 1

    print(test_labels_lstm.shape)


    # Now select a threshold
    threshold = 1000
    idx_detection = return_anomaly_idx_by_threshold(test_lstm_recons_error, threshold)
    # print(idx_detection)
    idx_detection_augmented = augment_detected_idx(idx_detection, anomaly_index_lstm)
    #print(anomaly_index_lstm)
    print(idx_detection_augmented)

    print("\nThreshold is {}".format(threshold))
    idx_detection = return_anomaly_idx_by_threshold(test_lstm_recons_error, threshold)
    idx_detection_augmented = augment_detected_idx(idx_detection, anomaly_index_lstm)

print("------START-TESTING-------")
start_testing=time.time()
resmon_process = subprocess.Popen(["resmon", "-o", "resmon_scada1_testing_1.csv"])
for i in range(len(test_data)):
    main_func(test_data[i])
resmon_process.terminate()
print("------END-TESTING-------")
end = time.time()
print("Total time: {}".format(end-start))
print("Testing time: {}".format(end-start_testing))
