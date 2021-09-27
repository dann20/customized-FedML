import json
import os
import argparse
from datetime import datetime

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
      config_dict = json.load(config_file)

    return config_dict

def save_config(config):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
    filename = config['result_dir'] + 'training_config_{}.json'.format(timestampStr)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def process_config(json_file):
    config = get_config_from_json(json_file)

    # create directories to save experiment results and trained models
    save_dir = "../VAE-XAI-related/experiments/{}/{}/batch-{}".format(
        config['exp_name'], config['dataset'], config['batch_size'])

    config['result_dir'] = os.path.join(save_dir, "result/")
    config['checkpoint_dir'] = os.path.join(save_dir, "checkpoint/")

    return config

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-n', '--num-client',
        type=int,
        default=4,
        help='The number of clients participating in Federated Learning')
    argparser.add_argument(
        '-ob', '--bmonOutfile',
        type=str,
        default='None',
        help='Bmon logfile')
    argparser.add_argument(
        '-or', '--resmonOutfile',
        type=str,
        default='None',
        help='Resmon logfile')
    args = argparser.parse_args()
    return args
