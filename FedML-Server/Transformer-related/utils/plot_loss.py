import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedavg.utils_Transformer import process_config

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config',
                           metavar='config',
                           default='None',
                           help='Init Configuration file')
    argparser.add_argument('-n', '--num-client',
                           type=int,
                           default=4,
                           help='The number of clients participating in Federated Learning')
    args = argparser.parse_args()
    return args

def plot_transformer(df_trans, config):
    for i in range(len(df_trans)):
        epoch = df_trans[i].loc[:,'CommRound']
        loss = df_trans[i].loc[:,'Loss']
        # Smooth plot
        x = np.linspace(epoch.min(), epoch.max(), 300)
        spl = make_interp_spline(epoch, loss)
        y = spl(x)
        plt.plot(x, y, label=f"Client {i+1}")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Transformer FL Client Loss - " + config["experiment"])
    plt.legend()
    plt.savefig(config['result_dir'] + config['experiment'] + '-transformer-client-loss' + '.png', dpi=300)

def plot_autoencoder(df_auto, config):
    for i in range(len(df_auto)):
        epoch = df_auto[i].loc[:,'Epoch']
        loss = df_auto[i].loc[:,'Loss']
        # Smooth plot
        x = np.linspace(epoch.min(), epoch.max(), 300)
        spl = make_interp_spline(epoch, loss)
        y = spl(x)
        plt.plot(x, y, label=f"Client {i+1}")
    plt.xlabel("Local Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder Client Loss - " + config["experiment"])
    plt.legend()
    plt.savefig(config['result_dir'] + config['experiment'] + '-autoencoder-client-loss' + '.png', dpi=300)

def main():
    try:
        args = get_args()
    except Exception as ex:
        print(ex)

    config = process_config(args.config)
    config['result_dir'] = config['result_dir'].replace("Transformer-related/", "") # Used with relative path
    print(config)
    client_dirs = [config['result_dir'] + f"client{i+1}/" for i in range(args.num_client)]
    df_auto = [pd.read_csv(client_dirs[i] + 'autoencoder_epoch_loss.csv') for i in range(len(client_dirs))]
    plot_autoencoder(df_auto, config)
    plt.clf()
    df_trans = [pd.read_csv(client_dirs[i] + 'transformer_epoch_loss.csv') for i in range(len(client_dirs))]
    plot_transformer(df_trans, config)

if __name__ == '__main__':
    main()
