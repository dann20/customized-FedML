import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedavg.utils_Transformer import process_config, get_config_from_json
from plot_loss import get_args

def main():
    try:
        args = get_args()
    except Exception as ex:
        print(ex)

    col_lst = ["experiment", "time", "auto_dataset",
               "validation", "num_client", "client_ID",
               "algorithm", "model", "d_model", "d_ff",
               "num_stacks", "num_heads", "dropout", "shuffle",
               "autoencoder_dims", "l_win", "pre_mask", "post_mask",
               "batch_size", "lr", "server_learning_rate",
               "L", "dataloader_num_workers",
               "auto_num_epoch", "num_comm_rounds", "trans_num_epoch",
               "auto_train_time", "total_trans_train_time",
               "q_best", "accuracy", "precision", "recall", "F1", "inference_time"]

    config = process_config(args.config)
    config['result_dir'] = config['result_dir'].replace("Transformer-related/", "") # Used with relative path
    print(config)
    if args.all:
        client_dirs = [config['result_dir'] + f"client{i+1}/" for i in range(args.num_client)]
    else:
        client_dirs = [config['result_dir'] + f"client{i+1}/" for i in args.client_id]
    print(client_dirs)
    result_file = "training_config_lwin_{}_autodims_{}.json".format(config["l_win"], config["autoencoder_dims"])
    result_dicts = [get_config_from_json(client_dirs[i] + result_file) for i in range(len(client_dirs))]
    df = pd.DataFrame(result_dicts)
    df = df.reindex(columns=col_lst)
    outfile = 'inference_results.csv'
    df.to_csv(outfile,
              mode='a',
              index=False,
              header=False if os.path.exists(outfile) else col_lst)

if __name__ == '__main__':
    main()
