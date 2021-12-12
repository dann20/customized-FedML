import logging
import os
import signal
import time

import requests

from .message_define import MyMessage

from FedML.fedml_core.distributed.communication.message import Message
from FedML.fedml_core.distributed.server.server_manager import ServerManager

class SCAFFOLDServerManager(ServerManager):
    def __init__(self, config, aggregator, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(config, comm, rank, size, backend)
        self.aggregator = aggregator
        self.num_rounds = config['num_comm_rounds']
        self.round_idx = 0
        self.num_client = size - 1

    def run(self):
        super().run()

    def finish(self):
        super().finish()
        time.sleep(10)
        response = requests.get('http://localhost:5000/shutdown')
        os.kill(os.getpid(), signal.SIGINT)

    def send_init_config(self):
        logging.info('sending init config...')
        self.round_idx = 0
        client_indexes = [self.round_idx] * self.num_client
        server_model = self.aggregator.get_server_model_params()
        server_controls = self.aggregator.get_server_control_variates()
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, server_model, server_controls, client_indexes[process_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        delta_model = msg_params.get(MyMessage.MSG_ARG_KEY_DELTA_MODEL_PARAMS)
        delta_controls = msg_params.get(MyMessage.MSG_ARG_KEY_DELTA_CONTROL_VARIATES)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.aggregator.add_local_trained_result(sender_id - 1, delta_model,
                                                 delta_controls, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            server_model, server_controls = self.aggregator.aggregate(self.round_idx)

            # start the next round
            self.round_idx += 1
            if self.round_idx < self.num_rounds:
                logging.info(f"-----START COMM ROUND {self.round_idx}-----")
                client_indexes = [self.round_idx]*self.num_client
                logging.info('indexes of clients: ' + str(client_indexes))
                logging.info("size = %d" % self.size)
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, server_model,
                                                           server_controls, client_indexes[receiver_id - 1])
            elif self.round_idx == self.num_rounds:
                self.finish()
                return

    def send_message_init_config(self, receive_id, server_model, server_controls, client_index):
        logging.info('sending init config to client ' + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, server_model)
        message.add_params(MyMessage.MSG_ARG_KEY_CONTROL_VARIATES, server_controls)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, server_model, server_controls, client_index):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, server_model)
        message.add_params(MyMessage.MSG_ARG_KEY_CONTROL_VARIATES, server_controls)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        self.send_message(message)
