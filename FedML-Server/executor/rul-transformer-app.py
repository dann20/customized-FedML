import logging
import os
import sys
import time
import subprocess
import atexit
import yaml
from datetime import datetime

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from FedML.fedml_api.distributed.fedavg.FedAvgTrainer_RULTransformer import FedAVGTransformerTrainer
from FedML.fedml_api.distributed.fedavg.FedAVGAggregator_RULTransformer import FedAVGAggregator
from FedML.fedml_api.distributed.fedavg.FedAvgServerManager_RULTransformer import FedAVGServerManager

from FedML.fedml_api.model.rul_transformer.rul_transformer import create_transformer

from FedML.fedml_api.distributed.fedavg.utils_RULTransformer import process_config, create_dirs, get_args, save_config
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


@app.route('/get-preprocessed-data/<dataset_name>', methods=['GET'])
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
    logging.info("Total running time: {} sec = {} min".format(run_time, run_time / 60))


if __name__ == '__main__':
    start_time = time.perf_counter()
    datetime_obj = datetime.now()

    if args.bmon:
        with open(args.bmon, 'w') as f:
            bmon_process = subprocess.Popen(['bmon', '-p', 'wlp7s0', '-r', '1', '-o', 'format:fmt=$(attr:txrate:bytes) $(attr:rxrate:bytes)\n'], stdout=f)
    else:
        bmon_process = None

    resmon_process = subprocess.Popen(["resmon", "-o", args.resmon]) if args.resmon else None
    tegrastats_process = subprocess.Popen(["tegrastats", "--logfile", args.tegrastats, "--interval", "1000"]) if args.tegrastats else None

    atexit.register(clean_subprocess, bmon_process, resmon_process, tegrastats_process, start_time)

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)

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
    logging.info(yaml.dump(config, default_flow_style=None))

    # create the experiments dirs
    create_dirs(config["result_dir"], config["checkpoint_dir"], config["server_model_dir"])
    # save the config in a json file in result directory
    save_config(config['result_dir'] + "result_lr_{}_l_win_{}_dff_{}.yml".format(
        config['lr'], config['l_win'], config['dff']), config)

    # wandb.init(
    #     project="fedml",
    #     name="mobile(mqtt)" + str(args.config),
    #     settings=wandb.Settings(start_method="fork"),
    #     config=args # needs attention
    # )

    size = args.num_client + 1

    # Set the random seed. torch.manual_seed determines the initial weight
    torch.manual_seed(10)

    model = create_transformer(d_model=config['d_model'],
                               nhead=config['n_head'],
                               dff=config['dff'],
                               num_layers=config['num_layers'],
                               dropout=config['dropout'],
                               l_win=config['l_win'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()

    trainer = FedAVGTransformerTrainer(id=0,
                                       transformer_model=model,
                                       train_data=None,
                                       criterion=criterion,
                                       optimizer=optimizer,
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

    server_manager.run()
    server_manager.send_init_config()

    # if run in debug mode, process will be single threaded by default
    app.run(host=cfg.APP_HOST, port=5000)
