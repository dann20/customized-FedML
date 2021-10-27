import argparse
import logging
import os
import sys
import time
from IPython import embed

import pandas as pd
import requests
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FedML.fedml_api.model.VAE_XAI.VAE_Model import VAEmodel
from FedML.fedml_api.data_preprocessing.VAE_XAI.data_loader import data_loader

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

    normal_train_data, _, test_data, test_labels, data = data_loader(config)

    vae_model = VAEmodel(config, "Client{}".format(client_ID))
    vae_model.load_model()
    vae_model.load_test_data(test_data, test_labels)
    vae_model.set_threshold(float(config["threshold"]))
    vae_model.test()

    # train = pd.DataFrame(normal_train_data)
    # test = pd.DataFrame(test_data)
    # shap.initjs()

    # shap_explainer = shap.KernelExplainer(vae_model.model,train[0:100])
    # shap_values = shap_explainer.shap_values(test[0:100])

    # shap.force_plot(shap_explainer.expected_value[3], shap_values[3])
    # shap.force_plot(shap_explainer.expected_value[0], shap_values[6][0], data.iloc[6, :])
    # shap.force_plot(shap_explainer.expected_value[0], shap_values[3][0], data.iloc[3, :])

    # embed()

    # shap.force_plot(shap_explainer.expected_value[0], shap_values[6][0], data.iloc[6, :], show=False, matplotlib=True).savefig('shape2.png')
    # time.sleep(1000000)
