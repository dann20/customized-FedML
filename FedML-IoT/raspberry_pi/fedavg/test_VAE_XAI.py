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
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    if config["load_dir"] == "default":
        normal_train_data, _, test_data, test_labels, data = data_loader('../VAE-XAI-related/datasets/dataset_processed.csv')
    else:
        normal_train_data, _, test_data, test_labels, data = data_loader(config["load_dir"])

    vae_model = VAEmodel(config, "Client{}".format(client_ID))
    vae_model.load_model()
    vae_model.set_threshold(0.15)
    vae_model.test(test_data, test_labels)

    train = pd.DataFrame(normal_train_data)
    test = pd.DataFrame(test_data)
    shap.initjs()

    shap_explainer = shap.KernelExplainer(vae_model.model,train[0:100])
    shap_values = shap_explainer.shap_values(test[0:100])

    shap.force_plot(shap_explainer.expected_value[3], shap_values[3])
    shap.force_plot(shap_explainer.expected_value[0], shap_values[6][0], data.iloc[6, :])
    shap.force_plot(shap_explainer.expected_value[0], shap_values[3][0], data.iloc[3, :])

    time.sleep(1000000)
