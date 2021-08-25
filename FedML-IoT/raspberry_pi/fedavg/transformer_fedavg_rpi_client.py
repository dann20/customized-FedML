# this script has been removed all unnecessary models and functions
import argparse
import logging
import os
import sys
import time
import subprocess
from datetime import datetime

import requests
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedavg.FedAvgClientManager_Transformer import FedAVGClientManager
from FedML.fedml_api.model.autoencoder.autoencoder import create_autoencoder
from FedML.fedml_api.model.transformer.transformer import create_transformer
from FedML.fedml_api.distributed.fedavg.Trainer_Autoencoder import AutoencoderTrainer
from FedML.fedml_api.distributed.fedavg.Trainer_Transformer import TransformerTrainer
from FedML.fedml_api.data_preprocessing.Transformer.data_loader import CustomDataset
from FedML.fedml_api.distributed.fedavg.utils_Transformer import process_config, create_dirs, get_args, save_config

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
    logging.info(main_args)
    logging.info("client_ID = " + str(client_ID))
    logging.info("experiment = " + str(config['experiment']))
    logging.info("dataset = " + str(config['auto_dataset']))

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
    config['time'] = timestampStr
    config['client_ID'] = client_ID
    logging.info(config)

    # create the experiments dirs
    create_dirs(config["result_dir"], config["checkpoint_dir"])
    # save the config in a json file in result directory
    save_config(config)

    # Set the random seed. torch_manual_seed determines the initial weight.
    # torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=bool(config["shuffle"]), num_workers=config["dataloader_num_workers"])

    autoencoder_model = create_autoencoder(in_seq_len=config['autoencoder_dims'],
                                           out_seq_len=config['l_win'],
                                           d_model=config['d_model'])
    autoencoder_trainer = AutoencoderTrainer(autoencoder_model=autoencoder_model,
                                             train_data=dataloader,
                                             device=device,
                                             config=config)
    autoencoder_trainer.train()
    config = autoencoder_trainer.get_updated_config()
    save_config(config)

    transformer_model = create_transformer(N=config['num_stacks'],
                                           d_model=config['d_model'],
                                           l_win=config['l_win'],
                                           d_ff=config['d_ff'],
                                           h=config['num_heads'],
                                           dropout=config['dropout'])
    transformer_trainer = TransformerTrainer(autoencoder_model=autoencoder_trainer.model,
                                             transformer_model=transformer_model,
                                             train_data=dataloader,
                                             device=device,
                                             config=config)

    size = config['num_client'] + 1
    client_manager = FedAVGClientManager(transformer_trainer,
                                         None,
                                         rank=client_ID,
                                         size=size,
                                         backend="MQTT",
                                         bmon_process=bmon_process,
                                         resmon_process=resmon_process)

    client_manager.run()

    time.sleep(1000000)
