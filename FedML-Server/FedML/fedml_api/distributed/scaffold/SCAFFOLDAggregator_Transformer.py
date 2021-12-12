import logging
import time

class SCAFFOLDAggregator(object):

    def __init__(self, transformer_trainer, num_clients):
        self.trainer = transformer_trainer
        self.num_clients = num_clients

        self.delta_model_dict = dict()
        self.delta_controls_dict = dict()
        self.sample_num_dict = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.num_clients):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_server_model_params(self):
        return self.trainer.get_model_params()

    def get_server_control_variates(self):
        return self.trainer.get_server_control_variates()

    def add_local_trained_result(self, index, delta_model, delta_controls, sample_num=None):
        logging.info("add_model. index = %d" % index)
        self.delta_model_dict[index] = delta_model
        self.delta_controls_dict[index] = delta_controls
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.num_clients):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.num_clients):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self, round_idx):
        start_time = time.perf_counter()

        logging.info(f"-----START SCAFFOLD AGGREGATION ROUND {round_idx}-----")
        for client in self.delta_controls_dict:
            for param, control, delta_control, delta_model in zip(self.trainer.model.parameters(),
                                                                  self.trainer.server_controls,
                                                                  self.delta_controls_dict[client],
                                                                  self.delta_model_dict[client]):
                param.data = param.data + self.trainer.config['server_learning_rate'] * delta_model.data / self.num_clients # (originally) num_of_selected_users
                control.data = control.data + delta_control.data / self.num_clients

        end_time = time.perf_counter()
        logging.info("-----DONE SCAFFOLD AGGREGATION-----")
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        if round_idx == self.trainer.config['num_comm_rounds'] - 1:
            self.trainer.save_aggregated_model(round_idx)
            logging.info("Saved last aggregated model.")

        return self.trainer.get_model_params(), self.trainer.get_server_control_variates()
