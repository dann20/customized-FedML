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
from .message_define import MyMessage

class FedAVGClientManager(ClientManager):
    def __init__(self, args, vae_model, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend) # now args is config_dict
        self.vae_model = vae_model
        self.num_rounds = args['num_comm_rounds']
        self.round_idx = 0

    def run(self):
        super().run()

    def finish(self):
        super().finish()
        os.kill(os.getpid(), signal.SIGINT)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_VAE_INIT_CONFIG,
                                              self.handle_message_vae_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_VAE_MODEL_TO_CLIENT,
                                              self.handle_message_receive_vae_model_from_server)

    def handle_message_vae_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        logging.info('received vae init')
        self.vae_model.set_vae_model_params(global_model_params)
        self.round_idx = 0
        self.__vae_train()

    def start_vae_training(self):
        self.round_idx = 0
        self.__vae_train()

    def handle_message_receive_vae_model_from_server(self, msg_params):
        logging.info("handle_message_receive_vae_model_from_server.")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        self.vae_model.set_vae_model_params(global_model_params)
        self.vae_model.save_model()
        self.round_idx += 1
        if self.round_idx < self.num_rounds:
            self.__vae_train()
        elif self.round_idx == self.num_rounds:
            self.finish()

    def send_vae_model_to_server(self, receive_id, vae_model_params):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_VAE_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_VAE_MODEL_PARAMS, vae_model_params)
        self.send_message(message)
        logging.info('sent vae model')

    def __vae_train(self):
        logging.info("#######VAE training########### round_id = %d" % self.round_idx)
        self.vae_model.train()
        local_train_vars = self.vae_model.get_vae_model_params()
        self.send_vae_model_to_server(0, local_train_vars)
