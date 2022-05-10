import logging
import os
import sys
import time
import subprocess
import atexit
import json
from datetime import datetime

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from FedML.fedml_api.distributed.fedavg.FedAvgTrainer_Transformer import FedAVGTransformerTrainer
from FedML.fedml_api.distributed.fedavg.FedAVGAggregator_Transformer import FedAVGAggregator
from FedML.fedml_api.distributed.fedavg.FedAvgServerManager_Transformer import FedAVGServerManager

from FedML.fedml_api.distributed.scaffold.SCAFFOLDTrainer_Transformer import SCAFFOLDTransformerTrainer
from FedML.fedml_api.distributed.scaffold.SCAFFOLDAggregator_Transformer import SCAFFOLDAggregator
from FedML.fedml_api.distributed.scaffold.SCAFFOLDServerManager_Transformer import SCAFFOLDServerManager

from FedML.fedml_api.model.transformer.transformer import create_transformer, create_fnet_hybrid

from FedML.fedml_api.distributed.fedavg.utils_Transformer import process_config, create_dirs, get_args, save_config
from FedML.fedml_iot import cfg

from FedML.fedml_core.distributed.communication.observer import Observer
from flask import Flask, request, jsonify, send_from_directory, abort

PASSWORD = "1"

# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'

# parse python script input parameters
try:
    args = get_args()
    config = process_config(args.config)
except Exception as ex:
    logging.error(ex)
    logging.error("Missing or invalid arguments")
    sys.exit(1)

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

    training_task_args = config
    training_task_args['num_client'] = args.num_client
    # training_task_args['dataset_url'] = '{}/get-preprocessed-data/{}'.format(request.url_root, client_id - 1)

    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "client_id": client_id,
                    "training_task_args": training_task_args})

@app.route("/shutdown", methods=['GET'])
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running werkzeug')
    shutdown_func()
    return "Shutting down..."

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

    if args.bmon:
        with open(args.bmon, 'w') as f:
            bmon_process = subprocess.Popen(['bmon', '-p', 'wlp7s0', '-r', '1', '-o', 'format:fmt=$(attr:txrate:bytes) $(attr:rxrate:bytes)\n'], stdout=f)
    else:
        bmon_process = None

    if args.resmon:
        resmon_process = subprocess.Popen(["resmon", "-o", args.resmon])
    else:
        resmon_process = None

    if args.tegrastats:
        tegrastats_process = subprocess.Popen(["tegrastats", "--logfile", args.tegrastats, "--interval", "1000"])
    else:
        tegrastats_process = None

    atexit.register(clean_subprocess, bmon_process, resmon_process, tegrastats_process, start_time)

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    timestamp = datetime_obj.strftime("%d-%b-%Y-%H:%M:%S")
    config['time'] = timestamp

    logging.info(args)
    logging.info(json.dumps(config, indent=4, separators=(',', ': ')))

    # create the experiments dirs
    create_dirs(config["result_dir"], config["checkpoint_dir"], config["server_model_dir"])
    # save the config in a json file in result directory
    save_config(config)

    # wandb.init(
    #     project="fedml",
    #     name="mobile(mqtt)" + str(args.config),
    #     settings=wandb.Settings(start_method="fork"),
    #     config=args # needs attention
    # )

    size = args.num_client + 1

    # Set the random seed. torch.manual_seed determines the initial weight
    torch.manual_seed(10)

    if config['model'] == 'transformer':
        transformer = create_transformer(N=config['num_stacks'],
                                         d_model=config['d_model'],
                                         l_win=config['l_win'],
                                         device=None,
                                         d_ff=config['d_ff'],
                                         h=config['num_heads'],
                                         dropout=config['dropout'])
    elif config['model'] == 'fnet_hybrid':
        transformer = create_fnet_hybrid(N=config['num_stacks'],
                                         d_model=config['d_model'],
                                         l_win=config['l_win'],
                                         device=None,
                                         d_ff=config['d_ff'],
                                         h=config['num_heads'],
                                         dropout=config['dropout'])
    else:
        logging.error("No valid model type specified in config file.")
        sys.exit(1)

    if config['algorithm'] == 'FedAvg':
        trainer = FedAVGTransformerTrainer(id = 0,
                                           autoencoder_model=None,
                                           transformer_model=transformer,
                                           train_data=None,
                                           val_data=None,
                                           device=None,
                                           config=config)
        aggregator = FedAVGAggregator(transformer_trainer=trainer,
                                      worker_num=args.num_client,
                                      client_weights=None)
        server_manager = FedAVGServerManager(config,
                                             aggregator,
                                             rank=0,
                                             size=size,
                                             backend="MQTT")
    elif config['algorithm'] == 'SCAFFOLD':
        trainer = SCAFFOLDTransformerTrainer(id = 0,
                                             autoencoder_model=None,
                                             transformer_model=transformer,
                                             train_data=None,
                                             val_data=None,
                                             device=None,
                                             config=config)
        aggregator = SCAFFOLDAggregator(transformer_trainer=trainer,
                                        num_clients=args.num_client)
        server_manager = SCAFFOLDServerManager(config,
                                               aggregator,
                                               rank=0,
                                               size=size,
                                               backend="MQTT")
    else:
        logging.error("No valid algorithm specified in config file.")
        sys.exit(1)

    server_manager.run()
    server_manager.send_init_config()

    # if run in debug mode, process will be single threaded by default
    app.run(host=cfg.APP_HOST, port=5000)
