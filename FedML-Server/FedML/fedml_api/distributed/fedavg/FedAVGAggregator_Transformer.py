import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

class FedAVGAggregator(object):

    def __init__(self, transformer_trainer, worker_num, client_weights=None):
        self.trainer = transformer_trainer
        self.worker_num = worker_num
        self.client_weights = client_weights

        self.model_dict = dict() # transformer models dict
        if not self.client_weights:
            self.sample_num_dict = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num=None):
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
            # self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx]) # no need due to byte-transfer instead of json
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            if not self.client_weights:
                training_num += self.sample_num_dict[idx]

        logging.info("length of self.model_dict = " + str(len(self.model_dict)))

        logging.info("-----START AGGREGATION-----")
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]

                if not self.client_weights:
                    w = local_sample_number / training_num
                else:
                    w = self.client_weights[i]

                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)
        logging.info("Set aggregated model to trainer.")
        self.trainer.save_aggregated_model()
        logging.info("Saved aggregated model.")
        end_time = time.time()
        logging.info("-----DONE AGGREGATION-----")
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params
