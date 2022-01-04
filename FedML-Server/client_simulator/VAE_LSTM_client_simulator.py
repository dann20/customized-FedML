# this script has been removed all unnecessary models and functions
import argparse
import logging
import os
import sys
import time
import json
from datetime import datetime

import requests
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FedML.fedml_api.distributed.fedavg.FedAvgClientManager_VAE_LSTM import FedAVGClientManager
from FedML.fedml_api.model.VAE_LSTM.VAE_LSTM_Models import VAEmodel
from FedML.fedml_api.distributed.fedavg.VAE_Trainer import vaeTrainer
from FedML.fedml_api.data_preprocessing.VAE_LSTM.data_loader import DataGenerator
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import create_dirs, save_config
from FedML.fedml_iot.cfg import APP_HOST

def add_args(parser):
    parser.add_argument('--client_uuid',
                        type=str,
                        default="0",
                        help='number of workers in a distributed cluster')
    args = parser.parse_args()
    return args


def register(uuid):
    str_device_UUID = uuid
    URL = "http://" + APP_HOST + ":5000/api/register"

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

def model_log(vae_trainer, lstm_model):
    print('----------- VAE MODEL ----------')
    vae_params = vae_trainer.get_vae_model_params()
    print('Len: ' + str(len(vae_params)))
    for i in range(len(vae_params)):
        print('Shape of layer ' + str(i) + str(vae_params[i].shape))

    print('----------- LSTM MODEL ----------')
    lstm_params = lstm_model.get_lstm_model_params()
    print('Len: ' + str(len(lstm_params)))
    for i in range(len(lstm_params)):
        print('Shape of layer ' + str(i) + str(lstm_params[i].shape))

if __name__ == '__main__':
    if sys.platform == 'darwin':
        # quick fix for issue : https://github.com/openai/spinningup/issues/16
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    client_ID, config = register(uuid)

    config["result_dir"] = os.path.join(config["result_dir"], "client{}/".format(client_ID))
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], "client{}/".format(client_ID))
    config["checkpoint_dir_lstm"] = os.path.join(config["checkpoint_dir_lstm"], "client{}/".format(client_ID))

    datetime_obj = datetime.now()
    timestamp = datetime_obj.strftime("%d-%b-%Y-%H:%M:%S")
    config['time'] = timestamp
    config['client_ID'] = client_ID
    config['dataset'] = config['dataset'] + '_' + str(client_ID)

    logging.info("experiment = " + str(config['exp_name']))
    logging.info("client_ID = " + str(client_ID))
    logging.info("dataset = " + str(config['dataset']))
    logging.info(json.dumps(config, indent=4, separators=(',', ': ')))

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())
    dataset = DataGenerator(config, client_ID+1)
    vae_model = VAEmodel(config, "Client{}".format(client_ID))
    vae_model.load(sess)
    vae_trainer = vaeTrainer(sess, vae_model, dataset, config)
    # lstm_model = lstmKerasModel("Client{}".format(client_ID), config)

    size = config['num_client'] + 1
    client_manager = FedAVGClientManager(config,
                                         vae_trainer,
                                         None,
                                         rank=client_ID,
                                         size=size,
                                         backend="MQTT")
    # model_log(client_manager.vae_trainer, client_manager.lstm_model)
    client_manager.run()
    # client_manager.start_training()

    time.sleep(1000000)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'False'
