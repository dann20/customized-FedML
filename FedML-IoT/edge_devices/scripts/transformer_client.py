"""
     This script has been removed all unnecessary models and functions.
     Used on both Raspberry Pi and Nvidia Jetson Nano.
"""

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
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

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

PASSWORD = "1"

def add_args(parser):
    parser.add_argument('--client_uuid',
                        type=str,
                        default="0",
                        help='number of workers in a distributed cluster')
    parser.add_argument('-b',
                        '--bmon',
                        type=str,
                        help='Bmon logfile')
    parser.add_argument('-r',
                        '--resmon',
                        type=str,
                        help='Resmon logfile')
    parser.add_argument('-t',
                        '--tegrastats',
                        type=str,
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

    if main_args.bmon:
        with open(main_args.bmon, 'w') as f:
            bmon_process = subprocess.Popen(['bmon', '-p', 'wlan0', '-r', '1', '-o', 'format:fmt=$(attr:txrate:bytes) $(attr:rxrate:bytes)\n'], stdout=f)
    else:
        bmon_process = None

    if main_args.resmon:
        resmon_process = subprocess.Popen(["resmon", "-o", main_args.resmon])
    else:
        resmon_process = None

    if main_args.tegrastats:
        echo_cmd = subprocess.Popen(['echo', PASSWORD], stdout=subprocess.PIPE)
        tegrastats_process = subprocess.Popen(["sudo", "-S", "tegrastats", "--logfile", main_args.tegrastats, "--interval", "1000"], stdin=echo_cmd.stdout)
    else:
        tegrastats_process = None

    atexit.register(clean_subprocess, bmon_process, resmon_process, tegrastats_process, start_time)

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    client_ID, config = register(uuid)
    logging.info(main_args)
    config['auto_dataset'] = config['auto_dataset'] + '_' + str(client_ID)
    logging.info("client_ID = " + str(client_ID))
    logging.info("experiment = " + str(config['experiment']))
    logging.info("dataset = " + str(config['auto_dataset']))

    timestamp = datetime_obj.strftime("%d-%b-%Y-%H:%M:%S")
    config['time'] = timestamp
    config['client_ID'] = client_ID
    logging.info(json.dumps(config, indent=4, separators=(',', ': ')))

    # create the experiments dirs
    create_dirs(config["result_dir"], config["checkpoint_dir"], config['server_model_dir'])
    # save the config in a json file in result directory
    save_config(config)

    size = config['num_client'] + 1

    # Set the random seed. torch_manual_seed determines the initial weight.
    torch.manual_seed(10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_initialized():
        torch.cuda.init()

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
