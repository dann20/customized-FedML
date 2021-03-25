import logging
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
except ImportError:
    from FedML.fedml_core.distributed.client.client_manager import ClientManager
    from FedML.fedml_core.distributed.communication.message import Message
from .message_define import MyMessage
from FedML.fedml_api.distributed.fedavg.utils_LCHA import to_list_arrays, to_nested_list


class FedAVGClientManager(ClientManager):
    def __init__(self, args, vae_trainer, lstm_model, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend) # now args is config_dict
        self.vae_trainer = vae_trainer
        self.lstm_model = lstm_model
        self.num_rounds = args['num_comm_rounds']
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_VAE_INIT_CONFIG,
                                              self.handle_message_vae_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_VAE_MODEL_TO_CLIENT,
                                              self.handle_message_receive_vae_model_from_server)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_LSTM_INIT_CONFIG,
                                              self.handle_message_lstm_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_LSTM_MODEL_TO_CLIENT,
                                              self.handle_message_receive_lstm_model_from_server)

    def handle_message_vae_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        global_model_params = to_list_arrays(global_model_params)

        self.vae_trainer.set_vae_model_params(global_model_params)
        self.round_idx = 0
        self.__vae_train()

    def start_training(self):
        self.round_idx = 0
        self.__vae_train()

    def handle_message_receive_vae_model_from_server(self, msg_params):
        logging.info("handle_message_receive_vae_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        global_model_params = to_list_arrays(global_model_params)
        self.vae_trainer.set_vae_model_params(global_model_params)
        self.round_idx += 1
        self.__vae_train()
        # if self.round_idx == self.num_rounds - 1:
        #     self.finish()

    def send_vae_model_to_server(self, receive_id, vae_model_params):
        vae_model_params = to_nested_list(vae_model_params)
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_VAE_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, vae_model_params)
        self.send_message(message)

    def __vae_train(self):
        logging.info("#######VAE training########### round_id = %d" % self.round_idx)
        vae_trainer.train()
        local_train_vars = vae_trainer.get_vae_model_params()
        self.send_vae_model_to_server(0, local_train_vars)

    def handle_message_lstm_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        global_model_params = to_list_arrays(global_model_params)

        self.lstm_model.set_lstm_model_params(global_model_params)
        self.round_idx = 0
        self.__lstm_train()

    def handle_message_receive_lstm_model_from_server(self, msg_params):
        logging.info("handle_message_receive_lstm_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        global_model_params = to_list_arrays(global_model_params)
        self.vae_trainer.set_lstm_model_params(global_model_params)
        self.round_idx += 1
        self.__lstm_train(global_model_params)
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_lstm_model_to_server(self, receive_id, lstm_model_params):
        lstm_model_params = to_nested_list(lstm_model_params)
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_VAE_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_LSTM_MODEL_PARAMS, lstm_model_params)
        self.send_message(message)

    def __lstm_train(self, global_model_params):
        logging.info("#######LSTM training########### round_id = %d" % self.round_idx)
        lstm_model.produce_embeddings(self.vae_trainer.model, self.vae_trainer.data, self.vae_trainer.sess)
        lstm_model.set_lstm_model_params(global_model_params)
        lstm_model.lstm_nn_model.summary()
        checkpoint_path = self.args['checkpoint_dir_lstm']\
                                          + "cp_{}.ckpt".format(lstm_model.name)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                          save_weights_only=True,
                                                          verbose=1)
        if self.args['lstm_epochs_per_comm_round'] > 0:
            lstm_model.train(lstm_model.lstm_nn_model, cp_callback)

        local_train_vars = lstm_model.get_lstm_model_params()
        self.send_lstm_model_to_server(0, local_train_vars)
