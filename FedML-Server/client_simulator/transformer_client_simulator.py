import argparse
import logging
import os
import sys
import time
import json
from datetime import datetime

import requests
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from FedML.fedml_api.distributed.fedavg.Trainer_Autoencoder import AutoencoderTrainer

from FedML.fedml_api.distributed.fedavg.FedAvgTrainer_Transformer import FedAVGTransformerTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgClientManager_Transformer import FedAVGClientManager

from FedML.fedml_api.distributed.scaffold.SCAFFOLDTrainer_Transformer import SCAFFOLDTransformerTrainer
from FedML.fedml_api.distributed.scaffold.SCAFFOLDClientManager_Transformer import SCAFFOLDClientManager

from FedML.fedml_api.data_preprocessing.Transformer.data_loader import CustomDataset
from FedML.fedml_api.model.autoencoder.autoencoder import create_autoencoder
from FedML.fedml_api.model.transformer.transformer import create_transformer, create_fnet_hybrid

from FedML.fedml_iot.cfg import APP_HOST
from FedML.fedml_api.distributed.fedavg.utils_Transformer import create_dirs, save_config

def add_args(parser):
    parser.add_argument('--client_uuid', type=str, default="0",
                        help='number of workers in a distributed cluster')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device used for torch training: "cpu" (default, fallback) or "gpu"')
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

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    # used when having multiple gpus in a single machine
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device

if __name__ == '__main__':
    if sys.platform == 'darwin':
        # quick fix for issue : https://github.com/openai/spinningup/issues/16
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    client_ID, config = register(uuid)
    config["result_dir"] = os.path.join(config["result_dir"], "client{}/".format(client_ID))
    config["checkpoint_dir"] = os.path.join(config["checkpoint_dir"], "client{}/".format(client_ID))
    config["auto_dataset"] = config["auto_dataset"] + "_" + str(client_ID) # each client will use a different file of dataset
    logging.info("client_ID = " + str(client_ID))
    logging.info("experiment = " + str(config['experiment']))
    logging.info("dataset = " + str(config['auto_dataset']))

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H:%M:%S")
    config['time'] = timestampStr
    config['client_ID'] = client_ID
    logging.info(json.dumps(config, indent=4, separators=(',', ': ')))

    # create the experiments dirs
    create_dirs(config['result_dir'], config['checkpoint_dir'], config['server_model_dir'])
    # save the config in a json file in result directory
    save_config(config)

    size = config['num_client'] + 1

    # Set the random seed. torch_manual_seed determines the initial weight.
    torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() and main_args.device == "gpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_initialized():
        torch.cuda.init()

    # device = init_training_device(client_ID - 1, args.client_num_per_round - 1, 4)

    train_set = CustomDataset(config, mode='train')
    train_dataloader = DataLoader(train_set,
                                  batch_size=config["batch_size"],
                                  shuffle=bool(config["shuffle"]),
                                  num_workers=config["dataloader_num_workers"])

    val_set = CustomDataset(config, mode='validate')
    if len(val_set) > 0:
        val_dataloader = DataLoader(val_set,
                                    batch_size=config["batch_size"],
                                    shuffle=bool(config["shuffle"]),
                                    num_workers=config["dataloader_num_workers"])
        config['validation'] = True
    else:
        val_dataloader = None
        config['validation'] = False

    autoencoder_model = create_autoencoder(in_seq_len=config['autoencoder_dims'],
                                           out_seq_len=config['l_win'],
                                           d_model=config['d_model'])
    autoencoder_model.float()
    autoencoder_trainer = AutoencoderTrainer(id = client_ID,
                                             autoencoder_model=autoencoder_model,
                                             train_data=train_dataloader,
                                             val_data=val_dataloader,
                                             device=device,
                                             config=config)
    autoencoder_trainer.train()
    config = autoencoder_trainer.get_updated_config()
    save_config(config)

    if config['model'] == 'transformer':
        transformer_model = create_transformer(N=config['num_stacks'],
                                               d_model=config['d_model'],
                                               l_win=config['l_win'],
                                               device=device,
                                               d_ff=config['d_ff'],
                                               h=config['num_heads'],
                                               dropout=config['dropout'])
    elif config['model'] == 'fnet_hybrid':
        transformer_model = create_fnet_hybrid(N=config['num_stacks'],
                                               d_model=config['d_model'],
                                               l_win=config['l_win'],
                                               device=device,
                                               d_ff=config['d_ff'],
                                               h=config['num_heads'],
                                               dropout=config['dropout'])
    else:
        logging.error("No valid model type specified in config file.")
        sys.exit(1)

    transformer_model.float()

    if config["algorithm"] == 'FedAvg':
        transformer_trainer = FedAVGTransformerTrainer(id = client_ID,
                                                       autoencoder_model=autoencoder_trainer.model,
                                                       transformer_model=transformer_model,
                                                       train_data=train_dataloader,
                                                       val_data=val_dataloader,
                                                       device=device,
                                                       config=config)

        client_manager = FedAVGClientManager(transformer_trainer,
                                             comm=None,
                                             rank=client_ID,
                                             size=size,
                                             backend="MQTT")
    elif config["algorithm"] == 'SCAFFOLD':
        transformer_trainer = SCAFFOLDTransformerTrainer(id = client_ID,
                                                         autoencoder_model=autoencoder_trainer.model,
                                                         transformer_model=transformer_model,
                                                         train_data=train_dataloader,
                                                         val_data=val_dataloader,
                                                         device=device,
                                                         config=config)
        client_manager = SCAFFOLDClientManager(transformer_trainer,
                                               comm=None,
                                               rank=client_ID,
                                               size=size,
                                               backend="MQTT")
    else:
        logging.error("No valid algorithm specified in config file.")
        sys.exit(1)

    client_manager.run()

    time.sleep(1000000)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'False'
