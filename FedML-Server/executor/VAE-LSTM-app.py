import logging
import os
import sys

import tensorflow as tf
import argparse
import numpy as np
import torch
import wandb
import pandas as pd

tf.compat.v1.disable_eager_execution()
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FedML.fedml_api.distributed.fedavg.FedAvgServerManager_VAE_LSTM import FedAVGServerManager
from FedML.fedml_api.distributed.fedavg.FedAVGAggregator_VAE_LSTM import FedAVGAggregator
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Models import VAEmodel, lstmKerasModel
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Trainer import vaeTrainer
from FedML.fedml_api.data_preprocessing.VAE_LSTM import DataGenerator
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import process_config, create_dirs, get_args, save_config

from FedML.fedml_core.distributed.communication.observer import Observer
from flask import Flask, request, jsonify, send_from_directory, abort

# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'

# parse python script input parameters
try:
    args = get_args()
    config = process_config(args.config)
except:
    print("missing or invalid arguments")
    exit(0)

device_id_to_client_id_dict = dict()


@app.route('/', methods=['GET'])
def index():
    return 'backend service for Fed_mobile'


@app.route('/get-preprocessed-data/<dataset_name>', methods = ['GET'])
def get_preprocessed_data(dataset_name):
    directory = app.config['MOBILE_PREPROCESSED_DATASETS'] + config['dataset'].upper() + '_mobile_zip/'
    try:
        return send_from_directory(
            directory,
            filename=dataset_name + '.zip',
            as_attachment=True)

    except FileNotFoundError:
        abort(404)


@app.route('/api/register', methods=['POST'])
def register_device():
    global device_id_to_client_id_dict
    # __log.info("register_device()")
    device_id = request.args['device_id']
    registered_client_num = len(device_id_to_client_id_dict)
    if device_id in device_id_to_client_id_dict:
        client_id = device_id_to_client_id_dict[device_id]
    else:
        client_id = registered_client_num + 1
        device_id_to_client_id_dict[device_id] = client_id

    training_task_args= {"exp_name": config[ 'exp_name' ],
                          "dataset": config[ 'dataset' ],
                          "y_scale": config[ 'y_scale '],
                          "one_image": config[ 'one_image' ],
                          "l_seq": config[ 'l_seq' ],
                          "l_win": config[ 'l_win' ],
                          "n_channel": config[ 'n_channel' ],
                          "TRAIN_VAE": config[ 'TRAIN_VAE' ],
                          "TRAIN_LSTM": config[ 'TRAIN_LSTM' ],
                          "TRAIN_sigma": config[ 'TRAIN_sigma' ],
                          "batch_size": config[ 'batch_size' ],
                          "batch_size_lstm": config[ 'batch_size_lstm' ],
                          "load_model": config[ 'load_model' ],
                          "load_dir": config[ 'load_dir' ],
                          "num_comm_rounds": config[ 'num_comm_rounds' ],
                          "vae_epochs_per_comm_round": config[ 'vae_epochs_per_comm_round'],
                          "lstm_epochs_per_comm_round": config[ 'lstm_epochs_per_comm_round' ],
                          "learning_rate_vae": config[ 'learning_rate_vae' ],
                          "learning_rate_lstm": config[ 'learning_rate_lstm' ],
                          "code_size": config[ 'code_size' ],
                          "sigma": config[ 'sigma' ],
                          "sigma2_offset": config[ 'sigma2_offset '],
                          "num_hidden_units": config[ 'num_hidden_units' ],
                          "num_hidden_units_lstm": config[ 'num_hidden_units_lstm' ],
                          'dataset_url': '{}/get-preprocessed-data/{}'.format(
                              request.url_root,
                              client_id-1
                              )
                          }

    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "client_id": client_id,
                    "training_task_args": training_task_args})

if __name__ == '__main__':
    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    logging.info(config)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    # save the config in a txt file
    save_config(config)

    wandb.init(
        project="fedml",
        name="mobile(mqtt)" + str(args.config),
        settings=wandb.Settings(start_method="fork"),
        config=args # needs attention
    )

    # create tensorflow session
    # sessions = []
    # data = []
    # model_vaes = []
    # vae_trainers = []
    # lstm_models = []
    model_vae_global = VAEmodel(config, "Global")
    sess_global = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())
    model_vae_global.load(sess_global)
    global_vae_trainer = vaeTrainer(sess_global, model_vae_global, None, config)
    global_lstm_model = lstmKerasModel("Global", config)
    client_weights = [0.1] * 8
    client_weights.append(0.2)
    aggregator = FedAVGAggregator(global_vae_trainer, global_lstm_model, args.num_client, config, client_weights)

    size = args.num_client + 1
    server_manager = FedAVGServerManager(config,
                                         aggregator,
                                         rank=0,
                                         size=size,
                                         backend="MQTT")
    server_manager.run()
    # if run in debug mode, process will be single threaded by default
    app.run(host='192.168.4.3', port=5000)
