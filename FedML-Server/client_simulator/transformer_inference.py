import argparse
import os
import sys
import math
import time
import json
import logging

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np
from scipy.stats import norm
from sklearn import metrics
from numba import jit
from numba.np.ufunc import parallel
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from FedML.fedml_api.model.autoencoder.autoencoder import create_autoencoder
from FedML.fedml_api.model.transformer.transformer import create_transformer, create_fnet_hybrid
from FedML.fedml_api.data_preprocessing.Transformer.data_loader import CustomDataset
from FedML.fedml_api.distributed.fedavg.utils_Transformer import save_config, get_config_from_json, process_config

torch.manual_seed(10)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and not torch.cuda.is_initialized():
    torch.cuda.init()


def add_args(parser):
    parser.add_argument("-c", "--config",
                        metavar="C",
                        default="None",
                        help="initial configuration file")
    parser.add_argument('-id', '--client-id',
                        type=int,
                        default=-1,
                        help='client_uuid, starting at index = 0')
    args = parser.parse_args()
    return args

def load_model(config):
    autoencoder_model = create_autoencoder(in_seq_len=config["autoencoder_dims"],
                                           out_seq_len=config["l_win"],
                                           d_model=config["d_model"])
    autoencoder_model.load_state_dict(torch.load(
        config["checkpoint_dir"] + "autoencoder_model.pt"))
    autoencoder_model.float()
    autoencoder_model.eval()

    if config['model'] == 'transformer':
        transformer_model = create_transformer(N=config["num_stacks"],
                                               d_model=config["d_model"],
                                               l_win=config["l_win"],
                                               device=device,
                                               d_ff=config["d_ff"],
                                               h=config["num_heads"],
                                               dropout=config["dropout"])
    elif config['model'] == 'fnet_hybrid':
        transformer_model = create_fnet_hybrid(N=config["num_stacks"],
                                               d_model=config["d_model"],
                                               l_win=config["l_win"],
                                               device=device,
                                               d_ff=config["d_ff"],
                                               h=config["num_heads"],
                                               dropout=config["dropout"])
    else:
        logging.error("No valid model type specified in config file.")
        sys.exit(1)

    transformer_model.load_state_dict(torch.load(
        config["checkpoint_dir"] + "transformer_model.pt"))
    transformer_model.float()
    transformer_model.eval()

    return autoencoder_model.encoder, transformer_model

def create_labels(idx_anomaly_test, n_test, config):
    anomaly_index = []
    test_labels = np.zeros(n_test)
    for i in range(len(idx_anomaly_test)):
        idx_start = idx_anomaly_test[i] - config["l_win"] + 1
        idx_end = idx_anomaly_test[i] + 1
        if idx_start < 0:
            idx_start = 0
        if idx_end > n_test:
            idx_end = n_test
        anomaly_index.append(np.arange(idx_start, idx_end))
        test_labels[idx_start:idx_end] = 1
    return anomaly_index, test_labels

def return_anomaly_idx_by_threshold(test_anomaly_metric, threshold):
    # test_list = np.squeeze(np.ndarray.flatten(test_anomaly_metric))
    idx_error = np.squeeze(np.argwhere(test_anomaly_metric > threshold))

    if len(idx_error.shape) == 0:
        idx_error = np.expand_dims(idx_error, 0)

    return list(idx_error)

@jit(parallel=True)
def augment_detected_idx(idx_detected_anomaly, anomaly_index):
    n_anomaly = len(anomaly_index)
    idx_detected_anomaly_extended = list(idx_detected_anomaly)
    for i in range(n_anomaly):
        # logging.info(idx_detected_anomaly)
        for j in idx_detected_anomaly:
            if j in anomaly_index[i]:
                in_original_detection = set(idx_detected_anomaly_extended)
                current_anomaly_win = set(anomaly_index[i])
                idx_detected_anomaly_extended = idx_detected_anomaly_extended + \
                    list(current_anomaly_win - in_original_detection)
                # logging.info(j)
                break
    return list(np.sort(idx_detected_anomaly_extended))

def count_TP_FP_FN(idx_detected_anomaly, test_labels):
    n_TP = 0
    n_FP = 0
    #n_detection = len(idx_detected_anomaly)
    # for i in range(n_detection):
    for i in idx_detected_anomaly:
        # if test_labels[idx_detected_anomaly[i]] == 1:
        if test_labels[i] == 1:
            n_TP = n_TP + 1
        else:
            n_FP = n_FP + 1

    idx_undetected = list(set(np.arange(len(test_labels))
                              ) - set(idx_detected_anomaly))
    n_FN = 0
    for i in idx_undetected:
        if test_labels[i] == 1:
            n_FN = n_FN + 1
    n_TN = len(test_labels) - n_TP - n_FP - n_FN
    return n_TP, n_FP, n_FN, n_TN

def compute_precision_and_recall(idx_detected_anomaly, test_labels):
    # compute true positive
    n_TP, n_FP, n_FN, n_TN = count_TP_FP_FN(idx_detected_anomaly, test_labels)

    if n_TP + n_FP == 0:
        precision = 1
    else:
        precision = n_TP / (n_TP + n_FP)
    recall = n_TP / (n_TP + n_FN)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall)/(precision + recall)
    accuracy = (n_TP + n_TN) / (n_TP + n_FN + n_FP + n_TN)

    return precision, recall, F1, accuracy, n_TP, n_FP, n_FN

def KQp(data, q):
    data2 = np.sort(data)  # sap xep tang dan
    n = np.shape(data2)[0]  # kich thuoc
    p = 1-q  # q tu xet, dat bang smth 0.05 0.025 0.01
    h = math.sqrt((p*q)/(n+1))
    KQ = 0
    for i in range(1, n+1):
        a = ((i/n)-p)/h
        b = (((i-1)/n)-p)/h
        TP = (norm.cdf(a)-norm.cdf(b))*data2[i-1]  # normcdf thu trong matlab
        KQ = KQ+TP
    # KQp = KQ;
    return KQ

def plot_roc_curve(fpr_aug, recall_aug, config, n_threshold=20):
    tpr = np.insert(recall_aug, [0, n_threshold], [0, 1])
    fpr = np.insert(fpr_aug, [0, n_threshold], [0, 1])
    auc = metrics.auc(fpr, tpr)
    logging.info("AUC =", auc)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw,
             label="ROC curve (area = %0.4f)" % auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Augmented Receiver operating characteristic of " +
              config["auto_dataset"])
    plt.legend(loc="lower right")
    plt.savefig(config["result_dir"] + "augmentedroc.pdf")
    return auc

def select_threshold(recon_loss, anomaly_index, test_labels, config, n_threshold=20):
    """
    Select best threshold based on best F1-score
    """
    precision = np.zeros(n_threshold)
    recall = np.zeros(n_threshold)
    F1 = np.zeros(n_threshold)
    precision_aug = np.zeros(n_threshold)
    recall_aug = np.zeros(n_threshold)
    F1_aug = np.zeros(n_threshold)
    fpr_aug = np.zeros(n_threshold)
    i = 0
    threshold_list = np.linspace(np.amin(recon_loss), np.amax(
        recon_loss), n_threshold, endpoint=False)
    threshold_list = np.flip(threshold_list)

    for threshold in threshold_list:
        # logging.info(threshold_list[i])
        idx_detection = return_anomaly_idx_by_threshold(recon_loss, threshold)
        # augment the detection using the ground truth labels
        # a method to discount the factor one anomaly appears in multiple consecutive windows
        # introduced in "Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications"
        idx_detection_augmented = augment_detected_idx(
            idx_detection, anomaly_index)
        precision_aug[i], recall_aug[i], F1_aug[i], fpr_aug[i], _, _, _ = compute_precision_and_recall(idx_detection_augmented,
                                                                                                       test_labels)
        i = i + 1
        #logging.info(precision, recall, F1)

    auc = plot_roc_curve(fpr_aug, recall_aug, config)

    logging.info("\nAugmented detection:")
    logging.info("Best F1 score is {}".format(np.amax(F1_aug)))
    idx_best_threshold = np.squeeze(np.argwhere(F1_aug == np.amax(F1_aug)))
    logging.info("Best threshold is {}".format(threshold_list[idx_best_threshold]))
    best_thres = np.min(threshold_list[idx_best_threshold])
    logging.info("At this threshold, precision is {}, recall is {}".format(precision_aug[idx_best_threshold],
                                                                    recall_aug[idx_best_threshold]))
    return best_thres, auc

def select_KQp_threshold(recon_loss, anomaly_index, test_labels, config):
    q_list = [0.99, 0.9, 0.1, 0.02, 0.01, 0.0095]
    n_threshold = len(q_list)
    precision_aug = np.zeros(n_threshold)
    recall_aug = np.zeros(n_threshold)
    F1_aug = np.zeros(n_threshold)
    acc_aug = np.zeros(n_threshold)
    q_best = 0
    for i in range(n_threshold):
        q = q_list[i]
        logging.info("Testing with q = {}".format(q))
        temp_thres = KQp(recon_loss, q)
        idx_detection = return_anomaly_idx_by_threshold(recon_loss, temp_thres)
        idx_detection_augmented = augment_detected_idx(
            idx_detection, anomaly_index)
        precision_aug[i], recall_aug[i], F1_aug[i], acc_aug[i], _, _, _ = compute_precision_and_recall(idx_detection_augmented, test_labels)
        logging.info("At this threshold, accuracy is {}, precision is {}, recall is {}, F1 is {}".format(acc_aug[i],
                                                                                                  precision_aug[i],
                                                                                                  recall_aug[i],
                                                                                                  F1_aug[i]))


    logging.info("Best F1 score is {}".format(max(F1_aug)))
    idx_best_q = np.argmax(F1_aug)
    q_best = q_list[idx_best_q]
    logging.info("Best q is {}".format(q_list[idx_best_q]))
    logging.info("At this threshold, accuracy is {}, precision is {}, recall is {}".format(acc_aug[idx_best_q],
                                                                                    precision_aug[idx_best_q],
                                                                                    recall_aug[idx_best_q]))

    # auc = plot_roc_curve(fpr_aug, recall_aug, config)
    return q_best, acc_aug[idx_best_q], precision_aug[idx_best_q], recall_aug[idx_best_q], F1_aug[idx_best_q]

def create_mask(config):
    mask = torch.ones(1, config["l_win"], config["l_win"])
    mask[:, config["pre_mask"]:config["post_mask"], :] = 0
    mask[:, :, config["pre_mask"]:config["post_mask"]] = 0
    mask = mask.float().masked_fill(mask == 0, float(
        "-inf")).masked_fill(mask == 1, float(0))
    return mask

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    try:
        args = add_args(parser)
        config = process_config(args.config)
    except Exception as ex:
        logging.error(ex)
        logging.error("Missing or invalid arguments")
        sys.exit(1)

    fmt = '[%(levelname)s] %(asctime)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format = fmt)

    # if model was trained by simulated clients, specify client_id (starting index = 0) to change result_dir
    if args.client_id != -1:
        config["result_dir"] = os.path.join(config["result_dir"], "client{}/".format(args.client_id + 1))
    filename = config["result_dir"] + \
        "training_config_lwin_{}_autodims_{}.json".format(
            config["l_win"], config["autoencoder_dims"])

    try:
        config = get_config_from_json(filename)
        logging.info(json.dumps(config, indent=4, separators=(',', ': ')))
    except Exception as ex:
        logging.error(ex)
        logging.error('No config found in result_dir.')

    dataset = CustomDataset(config, mode='test')
    dataloader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=bool(config["shuffle"]),
                            num_workers=config["dataloader_num_workers"])

    encoder, transformer = load_model(config)
    encoder.to(device)
    transformer.to(device)
    mask = create_mask(config)
    loss = torch.nn.MSELoss()
    n_test = len(dataset)
    logging.info(f"n_test = {n_test}")
    recon_loss = np.zeros(n_test)

    start = time.perf_counter()
    for i, batch in enumerate(dataloader):
        src = batch["input"].float().to(device)
        src = encoder(src)
        trg = batch["target"].float().to(device)
        trg = encoder(trg)
        out = transformer(src, src_mask=mask)
        for j in range(config["batch_size"]):
            try:
                recon_loss[i * config["batch_size"] + j] = loss(
                    out[j, config["pre_mask"]:config["post_mask"], :], trg[j, config["pre_mask"]:config["post_mask"], :])
            except:
                pass

    idx_anomaly_test = dataset.data["idx_anomaly_test"]
    anomaly_index, test_labels = create_labels(idx_anomaly_test,
                                               n_test,
                                               config)

    # Now select a threshold
    # threshold, auc = select_threshold(recon_loss,
    #                                   anomaly_index,
    #                                   test_labels,
    #                                   config)
    # config["AUC"] = auc

    # KQp_thres, q_best = select_KQp_threshold(threshold, recon_loss)
    # config["q_best"] = q_best
    # idx_detection = return_anomaly_idx_by_threshold(recon_loss, KQp_thres)
    # # logging.info(idx_detection)
    # idx_detection_augmented = augment_detected_idx(idx_detection, anomaly_index)
    # # logging.info(idx_detection_augmented)
    # precision, recall, F1, _, n_TP, n_FP, n_FN = compute_precision_and_recall(idx_detection_augmented,
                                                                              # test_labels)

    q_best, accuracy, precision, recall, F1 = select_KQp_threshold(recon_loss, anomaly_index, test_labels, config)
    config['q_best'] = q_best
    config['accuracy'] = accuracy
    config["precision"] = precision
    config["recall"] = recall
    config["F1"] = F1
    config["inference_time"] = (time.perf_counter() - start) / 60
    save_config(config)
    logging.info("PR evaluation using KQE:")
    logging.info("Accuracy: {}".format(accuracy))
    logging.info("Precision: {}".format(precision))
    logging.info("Recall: {}".format(recall))
    logging.info("F1: {}".format(F1))
    # logging.info("TP: {}".format(n_TP))
    # logging.info("FP: {}".format(n_FP))
    # logging.info("FN: {}".format(n_FN))

if __name__ == '__main__':
    main()
