# this script has been removed all unnecessary models and functions
import argparse
import logging
import os
import sys
import time
import subprocess
import atexit
import json
from datetime import datetime

import requests
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as ex:
        print(ex)

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 VAE-LSTM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FedML.fedml_api.distributed.fedavg.FedAvgClientManager_VAE_LSTM import FedAVGClientManager
from FedML.fedml_api.model.VAE_LSTM.VAE_LSTM_Models import VAEmodel
from FedML.fedml_api.distributed.fedavg.VAE_Trainer import vaeTrainer
from FedML.fedml_api.data_preprocessing.VAE_LSTM.data_loader import DataGenerator
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import create_dirs, save_config
from FedML.fedml_iot.cfg import APP_HOST

PASSWORD = "1"

def add_args(parser):
    parser.add_argument('--server_ip',
                        type=str,
                        default="http://127.0.0.1:5000",
                        help='IP address of the FedML server')
    parser.add_argument('--client_uuid',
                        type=str,
                        default="0",
                        help='number of workers in a distributed cluster')
    parser.add_argument('-b',
                        '--bmon',
                        type=str,
                        default='None',
                        help='Bmon logfile')
    parser.add_argument('-r',
                        '--resmon',
                        type=str,
                        default='None',
                        help='Resmon logfile')
    parser.add_argument('-t',
                        '--tegrastats',
                        type=str,
                        default='None',
                        help='tegrastats logfile')
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

def clean_subprocess(bmon_process, resmon_process, tegrastats_process, start_time):
    logging.info("Wait 10 seconds for server to end...")
    time.sleep(10)
    if bmon_process:
        bmon_process.terminate()
        logging.info("Terminated bmon.")
    if resmon_process:
        resmon_process.terminate()
        logging.info("Terminated resmon.")
    if tegrastats_process:
        echo_cmd = subprocess.Popen(['echo', PASSWORD], stdout=subprocess.PIPE)
        _ = subprocess.Popen(["sudo", "-S", "killall", "tegrastats"], stdin=echo_cmd.stdout)
        logging.info("Killed tegrastats.")
    run_time = time.perf_counter() - start_time
    logging.info("Total running time: {} sec = {} min".format(run_time, run_time/60))

if __name__ == '__main__':
    start_time = time.perf_counter()
    datetime_obj = datetime.now()
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    if main_args.bmon != 'None':
        bmon_command = "bmon -p wlan0 -r 1 -o 'format:fmt=$(attr:txrate:bytes) $(attr:rxrate:bytes)\n' > " + main_args.bmon
        bmon_process = subprocess.Popen(["exec " + bmon_command], shell=True)
    else:
        bmon_process = None

    if main_args.resmon!= 'None':
        resmon_process = subprocess.Popen(["resmon", "-o", main_args.resmon])
    else:
        resmon_process = None

    if main_args.tegrastats != 'None':
        echo_cmd = subprocess.Popen(['echo', PASSWORD], stdout=subprocess.PIPE)
        tegrastats_process = subprocess.Popen(["sudo", "-S", "tegrastats", "--logfile", main_args.tegrastats, "--interval", "1000"], stdin=echo_cmd.stdout)
    else:
        tegrastats_process = None

    atexit.register(clean_subprocess, bmon_process, resmon_process, tegrastats_process, start_time)

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    client_ID, config = register(uuid)

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

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=tf_config)
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
