import argparse
import logging
import os
import sys
import time

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FedML.fedml_api.distributed.fedavg.FedAvgClientManager_VAE import FedAVGClientManager
from FedML.fedml_api.model.VAE_XAI.VAE_Model import VAEmodel
from FedML.fedml_api.data_preprocessing.VAE_XAI.data_loader import data_loader
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import create_dirs, save_config
from FedML.fedml_api.model.svdd.svdd import SVDD

def add_args(parser):
    parser.add_argument('--client_uuid', type=str, default="0",
                        help='number of workers in a distributed cluster')
    args = parser.parse_args()
    return args

def register(uuid):
    str_device_UUID = uuid
    URL = "http://127.0.0.1:5000/api/register"

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

"""
python mobile_client_simulator.py --client_uuid '0'
python mobile_client_simulator.py --client_uuid '1'
"""
if __name__ == '__main__':
    if sys.platform == 'darwin':
        # quick fix for issue : https://github.com/openai/spinningup/issues/16
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    logging.basicConfig(level=logging.INFO)

    client_ID, config = register(uuid)
    config["result_dir"] = os.path.join(config["result_dir"], "client{}/".format(client_ID))
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], "client{}/".format(client_ID))
    config["dataset"] = config["dataset"] + "_" + str(client_ID)
    logging.info("experiment = " + str(config["exp_name"]))
    logging.info("client_ID = " + str(client_ID))
    logging.info("dataset = " + str(config['dataset']))
    logging.info(config)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir']])
    # save the config in a txt file
    save_config(config)

    normal_train_data, normal_val_data, test_data, test_labels, _ = data_loader(config)

    vae_model = VAEmodel(config, "Client{}".format(client_ID))
    vae_model.load_train_data(normal_train_data, normal_val_data)
    vae_model.load_test_data(test_data, test_labels)

    parameters = {"positive penalty": 0.9,
                  "negative penalty": 0.8,
                  "kernel": {"type": 'lapl', "width": 1/12},
                  "option": {"display": 'on'}}
    svdd = SVDD(parameters)

    if not config["multiple_thresholds"]:
        vae_model.set_threshold(float(config["threshold"]))
    else:
        vae_model.set_threshold(float(config["list_threshold"][client_ID-1]))
    logging.info("Set threshold = " + str(vae_model.threshold))

    size = config['num_client'] + 1
    client_manager = FedAVGClientManager(config,
                                         vae_model,
                                         svdd,
                                         rank=client_ID,
                                         size=size,
                                         backend="MQTT")

    client_manager.run()
    # client_manager.start_training()

    time.sleep(1000000)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'False'
