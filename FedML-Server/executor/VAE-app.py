import logging
import os
import sys
import subprocess
import time
import atexit

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedavg.FedAvgServerManager_VAE import FedAVGServerManager
from FedML.fedml_api.distributed.fedavg.FedAVGAggregator_VAE import FedAVGAggregator
from FedML.fedml_api.model.VAE_XAI.VAE_Model import VAEmodel
from FedML.fedml_api.distributed.fedavg.utils_VAE_XAI import process_config, create_dirs, get_args, save_config
from FedML.fedml_iot import cfg

from FedML.fedml_core.distributed.communication.observer import Observer
from flask import Flask, request, jsonify, send_from_directory, abort

# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'

# parse python script input parameters
try:
    args = get_args()
    config = process_config(args.config)
except Exception as ex:
    print(ex)
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

    training_task_args = config
    training_task_args['num_client'] = args.num_client

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

def model_log(vae_model):
    print('----------- VAE MODEL ----------')
    vae_params = vae_model.get_vae_model_params()
    print('Len: ' + str(len(vae_params)))
    for i in range(len(vae_params)):
        print('Shape of layer ' + str(i) + str(vae_params[i].shape))

def clean_subprocess(bmon_process, resmon_process, start_time):
    logging.info("Wait 10 seconds for server to end...")
    time.sleep(10)
    if bmon_process:
        bmon_process.terminate()
        logging.info("Terminated bmon.")
    if resmon_process:
        resmon_process.terminate()
        logging.info("Terminated resmon.")
    run_time = time.time() - start_time
    logging.info("Total running time: {} sec = {} min".format(run_time, run_time/60))

if __name__ == '__main__':
    start_time = time.time()
    if args.bmonOutfile != 'None':
        bmon_command = "bmon -p wlp7s0 -r 1 -o 'format:fmt=$(attr:txrate:bytes) $(attr:rxrate:bytes)\n' > " + args.bmonOutfile
        bmon_process = subprocess.Popen(["exec " + bmon_command], shell=True)
    else:
        bmon_process = None

    if args.resmonOutfile != 'None':
        resmon_process = subprocess.Popen(["resmon", "-o", args.resmonOutfile])
    else:
        resmon_process = None

    atexit.register(clean_subprocess, bmon_process, resmon_process, start_time)

    logging.basicConfig(level=logging.DEBUG)
    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    logging.info(config)

    # create the experiments dirs
    create_dirs([config['result_dir'], config['checkpoint_dir']])
    # save the config in a txt file
    save_config(config)

    # wandb.init(
    #     project="fedml",
    #     name="mobile(mqtt)" + str(args.config),
    #     settings=wandb.Settings(start_method="fork"),
    #     config=args # needs attention
    # )

    client_weights = [0.25, 0.25, 0.25, 0.25]
    global_vae_model = VAEmodel(config, "Global")
    aggregator = FedAVGAggregator(global_vae_model, args.num_client, config, client_weights)

    size = args.num_client + 1
    server_manager = FedAVGServerManager(config,
                                         aggregator,
                                         rank=0,
                                         size=size,
                                         backend="MQTT")
    # model_log(server_manager.aggregator.global_vae_model)
    server_manager.run()
    server_manager.send_init_vae_msg()

    # if run in debug mode, process will be single threaded by default
    app.run(host= cfg.APP_HOST, port=5000)
