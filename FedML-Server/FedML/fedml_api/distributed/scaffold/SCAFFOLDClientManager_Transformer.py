import logging
import os
import sys
import signal

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from FedML.fedml_api.distributed.fedavg.utils_Transformer import save_config
from .message_define import MyMessage

class SCAFFOLDClientManager(ClientManager):
    def __init__(self, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(trainer.config, comm, rank, size, backend)
        self.id = trainer.id
        self.trainer = trainer
        self.num_rounds = trainer.config['num_comm_rounds']
        self.round_idx = 0

    def run(self):
        super().run()

    def finish(self):
        super().finish()
        os.kill(os.getpid(), signal.SIGINT)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        server_control_variates = msg_params.get(MyMessage.MSG_ARG_KEY_CONTROL_VARIATES)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info('received init config')

        self.trainer.set_server_model_params(global_model_params)
        self.trainer.set_server_control_variates(server_control_variates)
        self.start_training()

    def start_training(self):
        self.round_idx = 0
        self.__train()
        self.__save_and_check()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        server_control_variates = msg_params.get(MyMessage.MSG_ARG_KEY_CONTROL_VARIATES)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.set_server_model_params(global_model_params)
        self.trainer.set_server_control_variates(server_control_variates)
        self.__train()
        self.__save_and_check()

    def send_delta_to_server(self, receive_id, delta_model, delta_controls, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_DELTA_MODEL_PARAMS, delta_model)
        message.add_params(MyMessage.MSG_ARG_KEY_DELTA_CONTROL_VARIATES, delta_controls)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)
        logging.info('sent transformer model')

    def __train(self):
        logging.info("#######TRANSFORMER TRAINING########### round_id = %d" % self.round_idx)
        self.trainer.train(self.round_idx)
        local_sample_num = self.trainer.num_train_samples
        logging.info(f'Client {self.id}. local_sample_num = {local_sample_num}')
        delta_model = self.trainer.get_delta_model_params()
        delta_controls = self.trainer.get_delta_control_variates()
        self.send_delta_to_server(0, delta_model, delta_controls, local_sample_num)

    def __save_and_check(self):
        save_config(self.trainer.config)
        self.round_idx += 1
        if self.round_idx == self.num_rounds:
            self.trainer.client_plot_loss()
            self.trainer.get_total_training_time(self.num_rounds)
            save_config(self.trainer.config)
            self.finish()
