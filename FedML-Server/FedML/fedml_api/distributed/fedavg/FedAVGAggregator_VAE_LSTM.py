import copy
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from FedML.fedml_api.data_preprocessing.VAE_LSTM.data_loader import \
    DataGenerator
from FedML.fedml_api.distributed.fedavg.utils_LCHA import (aa_to_list_arrays,
                                                           to_array_arrays,
                                                           to_list_arrays)
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import (create_dirs,
                                                               get_args,
                                                               process_config,
                                                               save_config)
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Models import (VAEmodel,
                                                                lstmKerasModel)
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Trainer import vaeTrainer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class FedAVGAggregator(object):

    def __init__(self, global_vae_trainer, global_lstm_model, worker_num, config, weights):
        self.global_lstm_model = global_lstm_model
        self.global_vae_trainer = global_vae_trainer
        self.config = config
        self.num_comm_rounds = config["num_comm_rounds"]
        self.weights = weights
        self.worker_num = worker_num

        # self.vae_model_dict = dict()
        self.train_vars_VAE_of_clients = list() # VAE model
        self.lstm_weights = list()
        # self.sample_num_dict = dict()

        self.flag_client_vae_model_uploaded_dict = dict()
        for idx in range(worker_num):
            self.flag_client_vae_model_uploaded_dict[idx] = False

        self.flag_client_lstm_model_uploaded_dict = dict()
        for idx in range(worker_num):
            self.flag_client_lstm_model_uploaded_dict[idx] = False

    def get_global_vae_model_params(self):
        global_train_var = list()
        for i in range(len(self.train_vars_VAE_of_clients[0])):
            global_train_var.append(self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess))
        return global_train_var

    def set_global_vae_model_params(self, global_train_var):
        for i in range(len(self.train_vars_VAE_of_clients[0])):
            self.global_vae_trainer.model.train_vars_VAE[i].load(global_train_var[i], self.global_vae_trainer.sess)

    def get_global_lstm_model_params(self):
        return self.global_lstm_model.lstm_nn_model.get_weights()

    def set_global_lstm_model_params(self, weights):
        self.global_lstm_model.lstm_nn_model.set_weights(weights)

    def add_vae_local_trained_result(self, index, model_params):
        logging.info("add_model. index = %d" % index)
        self.vae_model_dict[index] = model_params
        # self.sample_num_dict[index] = sample_num
        self.flag_client_vae_model_uploaded_dict[index] = True

    def add_lstm_local_trained_result(self, index, model_params):
        logging.info("add_model. index = %d" % index)
        self.lstm_model_dict[index] = model_params
        # self.sample_num_dict[index] = sample_num
        self.flag_client_lstm_model_uploaded_dict[index] = True

    def check_whether_all_receive_vae(self):
        for idx in range(self.worker_num):
            if not self.flag_client_vae_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_vae_model_uploaded_dict[idx] = False
        return True

    def check_whether_all_receive_lstm(self):
        for idx in range(self.worker_num):
            if not self.flag_client_lstm_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_lstm_model_uploaded_dict[idx] = False
        return True

    def run(self):
        if self.config['TRAIN_VAE'] and self.config['vae_epochs_per_comm_round'] > 0:
            self.aggregate_vae()
        if self.config['TRAIN_LSTM'] and self.config['lstm_epochs_per_comm_round'] > 0:
            self.aggregate_lstm()

    def aggregate_vae(self): # aggregate, set and save global VAE model
        start_time = time.time()
        global_train_var = list()
        for comm_round in range(self.config['num_comm_rounds']):
            # 2 parameters transmitted
            # [[self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess) for i in range(len(vae_trainer.model.train_vars_VAE))] for _ in range(len(num_clients))]
            # [ vae_trainer.model.train_vars_VAE for _ in range(num_clients)] # maybe no need
            for i in range(len(self.train_vars_VAE_of_clients[0])):
                global_train_var_eval = np.zeros_like(train_vars_VAE_of_clients[0][i])
                for client in range(len(self.train_vars_VAE_of_clients)):
                    global_train_var_eval += np.multiply(self.weights[client], self.train_vars_VAE_of_clients[client][i])
                print(global_train_var_eval)
                print('type of global_train_var_eval: ' + str( type(global_train_var_eval) ))
                self.global_vae_trainer.model.train_vars_VAE[i].load(global_train_var_eval, self.global_vae_trainer.sess) # set global vae model
                global_train_var.append(global_train_var_eval)

        end_time = time.time()
        logging.info("VAE aggregate time cost: %d" %(end_time - start_time))
        self.global_vae_trainer.model.save(self.global_vae_trainer.sess) # save global model
        return global_train_var

    def aggregate_lstm(self): # aggregate, set and save global LSTM model
        start_time = time.time()
        for comm_round in range(self.config['num_comm_rounds']):
            for i in range(len(self.lstm_weights)):
                if i == 0:
                    global_weights = np.multiply(self.lstm_weights[i], self.weights[i])
                else:
                    global_weights += np.multiply(self.lstm_weights[i], self.weights[i])
            self.global_lstm_model.lstm_nn_model.set_weights(global_weights) # set global lstm model
        end_time = time.time()
        logging.info("LSTM aggregate time cost: %d" %(end_time - start_time))
        glb_checkpoint_path = self.config['checkpoint_dir_lstm'] + "cp_{}.ckpt".format(self.global_lstm_model.name)
        self.global_lstm_model.lstm_nn_model.save_weights(glb_checkpoint_path)
        return global_weights
