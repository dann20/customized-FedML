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

    col_lst = ["experiment", "time", "client_ID", "batch_size", "num_stacks", "post_mask", "pre_mask",
                "autoencoder_dims", "l_win", "auto_num_epoch", "trans_num_epoch", "num_comm_rounds",
                "q_best", "AUC", "precision", "recall", "F1"]

    config = process_config(args.config)
    config['result_dir'] = config['result_dir'].replace("Transformer-related/", "") # Used with relative path
    print(config)
    client_dirs = [config['result_dir'] + f"client{i+1}/" for i in range(args.num_client)]
    print(client_dirs)
    result_file = "training_config_lwin_{}_autodims_{}.json".format(config["l_win"], config["autoencoder_dims"])
    result_dicts = [get_config_from_json(client_dirs[i] + result_file) for i in range(len(client_dirs))]
    df = pd.DataFrame(result_dicts)
    df = df[col_lst]
    outfile = 'inference_results.csv'
    df.to_csv(outfile,
              mode='a',
              index=False,
              header=False if os.path.exists(outfile) else col_lst)

if __name__ == '__main__':
    main()
