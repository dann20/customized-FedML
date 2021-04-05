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

class FedAVGServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.aggregator = aggregator
        self.round_num = args['num_comm_rounds']
        self.round_idx = 0
        self.num_client = size - 1
        self.phase_confirm = dict()
        for idx in range(self.num_client):
            self.phase_confirm[idx] = False

    def run(self):
        super().run()

    def send_init_vae_msg(self):
        logging.info('sending init VAE model...')
        self.round_idx = 0
        client_indexes = [self.round_idx]*self.num_client
        global_model_params = self.aggregator.get_global_vae_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_vae_config(process_id, global_model_params, client_indexes[process_id - 1])
        logging.info('all init vae msgs sent')

    def send_init_lstm_msg(self):
        logging.info('sending init LSTM model...')
        self.round_idx = 0
        client_indexes = [self.round_idx]*self.num_client
        global_model_params = self.aggregator.get_global_lstm_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_lstm_config(process_id, global_model_params, client_indexes[process_id - 1])
        logging.info('all init lstm msgs sent')

    def send_init_config(self):
        logging.info('sending init config...')
        self.round_idx = 0
        client_indexes = [self.round_idx] * self.num_client
        global_vae_params = self.aggregator.get_global_vae_model_params()
        global_lstm_params = self.aggregator.get_global_lstm_model_params()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_vae_params, global_lstm_params, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_VAE_MODEL_TO_SERVER,
                                              self.handle_message_receive_vae_model_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_LSTM_MODEL_TO_SERVER,
                                              self.handle_message_receive_lstm_model_from_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_ACTIVATE_LSTM_PHASE,
                                              self.handle_message_phase_confirm_from_client)

    def check_whether_all_phase_confirm(self):
        for idx in range(self.num_client):
            if not self.phase_confirm[idx]:
                return False
        for idx in range(self.num_client):
            self.phase_confirm[idx] = False
        return True

    def handle_message_phase_confirm_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        self.phase_confirm[sender_id - 1] = True
        logging.info('received phase confirm from client ' + str(sender_id))
        b_all_received = self.check_whether_all_phase_confirm()
        logging.info('b_phase_confirm_all_received = ' + str(b_all_received))
        if b_all_received:
            self.send_init_config()
            logging.info('received all connection confirmations and sent init model')

    def handle_message_receive_vae_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        vae_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS)
        self.aggregator.add_vae_local_trained_result(sender_id - 1, vae_model_params)
        # logging.info(self.aggregator.train_vars_VAE_of_clients[sender_id-1])
        logging.info('received vae model from client ' + str(sender_id))
        b_all_received = self.aggregator.check_whether_all_receive_vae()
        logging.info("b_vae_all_received = " + str(b_all_received))
        if b_all_received:
            global_vae_model_params = self.aggregator.aggregate_vae()
            self.round_idx += 1
            client_indexes = [self.round_idx]*self.num_client
            print('indexes of clients: ' + str(client_indexes))
            print('size = %d' % self.size)
            for receiver_id in range(1,self.size):
                self.send_message_sync_vae_model_to_client(receiver_id, global_vae_model_params, client_indexes[receiver_id - 1])

    def handle_message_receive_lstm_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        lstm_global_params = msg_params.get(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS)
        self.aggregator.add_lstm_local_trained_result(sender_id - 1, lstm_global_params)
        logging.info('received lstm model from client ' + str(sender_id))
        b_all_received = self.aggregator.check_whether_all_receive_lstm()
        logging.info("b_lstm_all_received = " + str(b_all_received))
        if b_all_received:
            global_lstm_model_params = self.aggregator.aggregate_lstm()
            self.round_idx += 1
            client_indexes = [self.round_idx]*self.num_client
            print('indexes of clients: ' + str(client_indexes))
            print('size = %d' % self.size)
            for receiver_id in range(1,self.size):
                self.send_message_sync_lstm_model_to_client(receiver_id, global_lstm_model_params, client_indexes[receiver_id - 1])
        if self.round_idx == self.round_num:
            self.finish()

    def send_message_init_vae_config(self, receive_id, global_model_params, client_index):
        logging.info('sent init vae to client ' + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_VAE_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_init_lstm_config(self, receive_id, global_model_params, client_index):
        logging.info('sent init lstm to client ' + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_LSTM_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_init_config(self, receive_id, global_vae_params, global_lstm_params, client_index):
        logging.info('sending init config to client ' + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, global_vae_params)
        message.add_params(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS, global_lstm_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_vae_model_to_client(self, receive_id, global_vae_model_params, client_index):
        logging.info("send_message_sync_vae_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_VAE_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, global_vae_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_lstm_model_to_client(self, receive_id, global_lstm_model_params, client_index):
        logging.info("send_message_sync_lstm_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_LSTM_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS, global_lstm_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
