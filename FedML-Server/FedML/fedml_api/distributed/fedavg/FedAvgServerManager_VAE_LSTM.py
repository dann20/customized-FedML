import logging
import os
import sys

from .message_define import MyMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
try:
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedML.fedml_core.distributed.communication.message import Message
    from FedML.fedml_core.distributed.server.server_manager import ServerManager
from FedML.fedml_api.distributed.fedavg.utils_LCHA import to_nested_list

class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.num_client = size - 1

    def run(self):
        super().run()

    def send_init_vae_msg(self):
        global_model_params = self.aggregator.get_global_vae_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_vae_config(process_id, global_model_params, process_id - 1)

    def send_init_lstm_msg(self):
        global_model_params = self.aggregator.get_global_lstm_model_params()
        global_model_params = to_nested_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_lstm_config(process_id, global_model_params, process_id - 1)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        if self.round_idx <= self.round_num:
            sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
            vae_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS)
            self.aggregator.add_vae_local_trained_result(sender_id - 1, vae_model_params)
            b_all_received = self.aggregator.check_whether_all_receive_vae()
            logging.info("b_all_received = " + str(b_all_received))
            if b_all_received:
                global_vae_model_params = self.aggregator.aggregate_vae()
                self.round_idx += 1
                client_indexes = [self.round_idx]*self.num_client
                print('indexes of clients: ' + str(client_indexes))
                print('size = %d' % self.size)
                print("transforming to transmit-able data...")
                global_vae_model_params = to_nested_list(global_vae_model_params)
                for receiver_id in range(1,self.size):
                    self.send_message_sync_vae_model_to_client(receiver_id, global_vae_model_params, client_indexes[receiver_id - 1])
        elif self.round_idx <= (self.round_num * 2):
            sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
            lstm_global_params = msg_params.get(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS)
            self.aggregator.add_lstm_local_trained_result(sender_id - 1, lstm_global_params)
            b_all_received = self.aggregator.check_whether_all_receive_lstm()
            logging.info("b_all_received = " + str(b_all_received))
            if b_all_received:
                global_lstm_model_params = self.aggregator.aggregate_lstm()
                self.round_idx += 1
                client_indexes = [self.round_idx]*self.num_client
                print('indexes of clients: ' + str(client_indexes))
                print('size = %d' % self.size)
                print("transforming to transmit-able data...")
                global_lstm_model_params = to_nested_list(global_lstm_model_params)
                for receiver_id in range(1,self.size):
                    self.send_message_sync_lstm_model_to_client(receiver_id, global_lstm_model_params, client_indexes[receiver_id - 1])
        else:
            self.finish()

    def send_message_init_vae_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_init_lstm_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_vae_model_to_client(self, receive_id, global_vae_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, global_vae_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_lstm_model_to_client(self, receive_id, global_lstm_model_params, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS, global_lstm_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
