import copy
import logging
import time

import numpy as np
import wandb

from FedML.fedml_api.distributed.fedavg.utils_LCHA import to_list_arrays, to_array_arrays, aa_to_list_arrays

class FedAVGAggregator2(object):

    def __init__(self, train_pca, train_label, test_pca, test_label, worker_num, args, model_trainer):
        self.trainer = model_trainer

        self.train_pca = train_pca
        self.train_label = train_label
        self.test_pca = test_pca
        self.test_label = test_label

        self.worker_num = worker_num
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            self.model_dict[idx] = to_list_arrays(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (_, averaged_params) = model_list[0]
        if len(model_list) > 1:
            for i in range(1, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                averaged_params += np.multiply(local_model_params, w, dtype=object)

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server_for_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
            # test on train dataset
            train_loss, train_acc = self.trainer.test(self.train_pca, self.train_label)
            print('Done testing train data')
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {'train_acc': train_acc, 'train_loss': train_loss}
            logging.info(stats)

            # test on test dataset
            test_loss, test_acc = self.trainer.test(self.test_pca, self.test_label)
            print('Done testing test data')
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
