# this script has been removed all unnecessary models and functions
import argparse
import logging
import os
import sys
import time
import subprocess

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FedML.fedml_api.distributed.fedavg.FedAvgClientManager_VAE import FedAVGClientManager
from FedML.fedml_api.model.VAE_XAI.VAE_Model import VAEmodel
from FedML.fedml_api.data_preprocessing.VAE_XAI.data_loader import data_loader
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import process_config_VAE, create_dirs, get_args, save_config

from FedML.fedml_core.distributed.communication.observer import Observer

def add_args(parser):
    parser.add_argument('--server_ip',
                        type=str,
                        default="http://127.0.0.1:5000",
                        help='IP address of the FedML server')
    parser.add_argument('--client_uuid',
                        type=str,
                        default="0",
                        help='number of workers in a distributed cluster')
    parser.add_argument('-ob',
                        '--bmonOutfile',
                        type=str,
                        default='None',
                        help='Bmon logfile')
    parser.add_argument('-or',
                        '--resmonOutfile',
                        type=str,
                        default='None',
                        help='Resmon logfile')
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
    config = result['training_task_args']

    return client_ID, config

def model_log(vae_model):
    print('----------- VAE MODEL ----------')
    vae_params = vae_model.get_vae_model_params()
    print('Len: ' + str(len(vae_params)))
    for i in range(len(vae_params)):
        print('Shape of layer ' + str(i) + str(vae_params[i].shape))

"""
python mobile_client_simulator.py --client_uuid '0'
python mobile_client_simulator.py --client_uuid '1'
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    if main_args.bmonOutfile != 'None':
        bmon_command = "bmon -p wlan0 -r 1 -o 'format:fmt=$(attr:txrate:bytes) $(attr:rxrate:bytes)\n' > " + main_args.bmonOutfile
        bmon_process = subprocess.Popen([bmon_command], shell=True)
    else:
        bmon_process = None

    if main_args.resmonOutfile != 'None':
        resmon_process = subprocess.Popen(["resmon", "-o", main_args.resmonOutfile])
    else:
        resmon_process = None

    logging.basicConfig(level=logging.INFO)

    client_ID, config = register(main_args, uuid)
    logging.info("client_ID = " + str(client_ID))
    logging.info("dataset = " + str(config['dataset']))
    logging.info(config)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir']])
    # save the config in a txt file
    save_config(config)

    if config["load_dir"] == "default":
        normal_train_data, normal_test_data, _, _, _ = data_loader('../VAE-XAI-related/datasets/dataset_processed.csv')
    else:
        normal_train_data, normal_test_data, _, _, _ = data_loader(config["load_dir"])

    vae_model = VAEmodel(config, "Client{}".format(client_ID))
    vae_model.load_train_data(normal_train_data, normal_test_data)

    size = config['num_client'] + 1
    client_manager = FedAVGClientManager(config,
                                         vae_model,
                                         rank=client_ID,
                                         size=size,
                                         backend="MQTT",
                                         bmon_process=bmon_process,
                                         resmon_process=resmon_process)
    # model_log(client_manager.vae_model)
    client_manager.run()
    # client_manager.start_training()

    time.sleep(1000000)
