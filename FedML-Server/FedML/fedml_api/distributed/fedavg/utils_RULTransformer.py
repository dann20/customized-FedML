import argparse
import yaml
import os


def get_config_from_yaml(PATH):
    with open(PATH, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def save_config(PATH, config):
    with open(PATH, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def create_dirs(*dirs):
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


def process_config(yaml_file):
    config = get_config_from_yaml(yaml_file)
    save_dir = f"../RUL-related/experiments/{config['experiment']}/"
    config["result_dir"] = os.path.abspath(os.path.join(save_dir, "results/")) + "/"
    config["server_model_dir"] = os.path.abspath(os.path.join(save_dir, "server_models/")) + "/"
    config["data_path"] = os.path.abspath("../RUL-related/datasets/") + "/"
    config["checkpoint_dir"] = os.path.abspath(os.path.join(save_dir, "checkpoints/")) + "/"
    return config


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='YAML configuration file')
    argparser.add_argument(
        '-n', '--num-client',
        type=int,
        default=4,
        help='The number of clients participating in Federated Learning')
    argparser.add_argument(
        '-b', '--bmon',
        type=str,
        help='Bmon logfile')
    argparser.add_argument(
        '-r', '--resmon',
        type=str,
        help='Resmon logfile')
    argparser.add_argument(
        '-t', '--tegrastats',
        type=str,
        help='tegrastats logfile')
    args = argparser.parse_args()
    return args
