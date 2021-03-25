# this script has been removed all unnecessary models and functions
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA

import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from FedML.fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer
from FedML.fedml_api.distributed.fedavg.MyModelTrainer_LCHA import MyModelTrainerLCHA
from FedML.fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager
from FedML.fedml_api.distributed.fedavg.FedAVGTrainer2 import FedAVGTrainer2
from FedML.fedml_api.distributed.fedavg.FedAvgClientManager2 import FedAVGClientManager2

from FedML.fedml_api.data_preprocessing.LCHA_xlsx.data_loader import load_data_LCHA

from FedML.fedml_api.model.LCHA.LCHA import LCHA


def add_args(parser):
    parser.add_argument('--server_ip', type=str, default="http://127.0.0.1:5000",
                        help='IP address of the FedML server')
    parser.add_argument('--client_uuid', type=str, default="0",
                        help='number of workers in a distributed cluster')
    args = parser.parse_args()
    return args


def register(args, uuid):
    str_device_UUID = uuid
    URL = args.server_ip + "/api/register"

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'device_id': str_device_UUID}

    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS)
    result = r.json()
    client_ID = result['client_id']
    # executorId = result['executorId']
    # executorTopic = result['executorTopic']
    training_task_args = result['training_task_args']

    class Args:
        def __init__(self):
            self.dataset = training_task_args['dataset']
            self.data_dir = training_task_args['data_dir']
            self.partition_method = training_task_args['partition_method']
            self.partition_alpha = training_task_args['partition_alpha']
            self.model = training_task_args['model']
            self.client_num_per_round = training_task_args['client_num_per_round']
            self.comm_round = training_task_args['comm_round']
            self.epochs = training_task_args['epochs']
            self.lr = training_task_args['lr']
            self.wd = training_task_args['wd']
            self.batch_size = training_task_args['batch_size']
            self.frequency_of_the_test = training_task_args['frequency_of_the_test']
            self.is_mobile = training_task_args['is_mobile']

    args = Args()
    return client_ID, args

def load_data(args, dataset_name):
    logging.info("load_data. dataset_name = %s" % dataset_name)
    train_data, train_label, \
    test_data, test_label, class_num = load_data_LCHA(train_data_dir = "./../../FedML/data/LCHA_xlsx/train",
                                           test_data_dir = "./../../FedML/data/LCHA_xlsx/test")
    dataset = [train_data, train_label, test_data, test_label, class_num]
    return dataset

def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = LCHA().build(6,output_dim) # nk=6
    return model


"""
python mobile_client_simulator.py --client_uuid '0'
python mobile_client_simulator.py --client_uuid '1'
"""
if __name__ == '__main__':
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    client_ID, args = register(main_args, uuid)
    logging.info("client_ID = " + str(client_ID))
    logging.info("dataset = " + str(args.dataset))
    logging.info("model = " + str(args.model))
    logging.info("client_num_per_round = " + str(args.client_num_per_round))
    client_index = client_ID - 1

    dataset = load_data(args, args.dataset)
    logging.info("client_ID = %d, size = %d" % (client_ID, args.client_num_per_round))
    [train_data, train_label, test_data, test_label, class_num] = dataset
    model = create_model(args, model_name=args.model, output_dim=dataset[4])
    model_trainer = MyModelTrainerLCHA(model,args)
    model_trainer.set_id(client_index)
    # PCA transform
    pca_model = 'pca.sav'
    pca = pickle.load(open(pca_model, 'rb'))
    train_pca = pca.transform(train_data)
    test_pca = pca.transform(test_data)

    # start training
    trainer = FedAVGTrainer2(client_index, train_pca, train_label, test_pca, test_label, args, model_trainer)

    size = args.client_num_per_round + 1
    client_manager = FedAVGClientManager2(args, trainer, rank=client_ID, size=size, backend="MQTT")
    client_manager.run()
    client_manager.start_training()

    time.sleep(100000)
