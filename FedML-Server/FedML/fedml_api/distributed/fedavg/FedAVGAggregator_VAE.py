import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

class FedAVGAggregator(object):

    def __init__(self, global_vae_model, worker_num, config, weights):
        self.global_vae_model = global_vae_model
        self.config = config
        self.num_comm_rounds = config["num_comm_rounds"]
        self.weights = weights
        self.worker_num = worker_num

        self.vae_model_dict = dict() # VAE model

        self.flag_client_vae_model_uploaded_dict = dict()
        for idx in range(worker_num):
            self.flag_client_vae_model_uploaded_dict[idx] = False

    def get_global_vae_model_params(self):
        return self.global_vae_model.get_vae_model_params()

    def set_global_vae_model_params(self, weights): # arg is list of arrays
        self.global_vae_model.set_vae_model_params(weights)

    def add_vae_local_trained_result(self, index, model_params):
        logging.info("add_VAE_model. index = %d" % index)
        self.vae_model_dict[index] = model_params
        self.flag_client_vae_model_uploaded_dict[index] = True

    def check_whether_all_receive_vae(self):
        for idx in range(self.worker_num):
            if not self.flag_client_vae_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_vae_model_uploaded_dict[idx] = False
        return True

    def aggregate_vae(self): # aggregate, set and save global VAE model
        logging.info("start aggregating VAE model...")
        start_time = time.time()
        for i in range(len(self.vae_model_dict)):
            if i == 0:
                global_weights = np.multiply(self.vae_model_dict[i], self.weights[i])
            else:
                global_weights += np.multiply(self.vae_model_dict[i], self.weights[i])
        end_time = time.time()
        logging.info("VAE aggregate time cost: %d" %(end_time - start_time))
        logging.info("done aggregate_vae function")
        return global_weights # return list of arrays
