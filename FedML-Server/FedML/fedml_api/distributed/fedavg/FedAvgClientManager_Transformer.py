import logging
import os
import sys

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

class FedAVGClientManager(ClientManager):
    def __init__(self, trainer, comm=None, rank=0, size=0, backend="MPI", bmon_process=None, resmon_process=None):
        super().__init__(trainer.config, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = trainer.config['num_comm_rounds']
        self.round_idx = 0
        if bmon_process:
            self.bmon_process = bmon_process
        if resmon_process:
            self.resmon_process = resmon_process

    def run(self):
        super().run()

    # def finish(self):
    #     super().finish()
    #     if self.bmon_process:
    #         self.bmon_process.terminate()
    #     if self.resmon_process:
    #         self.resmon_process.terminate()
    #     sys.exit()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info('received init config')

        self.trainer.set_model_params(global_model_params)
        self.start_training()

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.trainer.set_model_params(global_model_params)
        self.trainer.save_aggregated_model()
        self.round_idx += 1
        if self.round_idx < self.num_rounds:
            self.__train()
            save_config(self.trainer.config)
        if self.round_idx == self.num_rounds:
            save_config(self.trainer.config)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)
        logging.info('sent transformer model')

    def __train(self):
        logging.info("#######TRANSFORMER TRAINING########### round_id = %d" % self.round_idx)
        self.trainer.train(self.round_idx)
        local_sample_num = self.trainer.get_len_data()
        logging.info(f'local_sample_num = {local_sample_num}')
        weights = self.trainer.get_model_params()
        self.send_model_to_server(0, weights, local_sample_num)
