import json
import sys
import pickle
import logging

class Message(object):

    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    # MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_VAE_MODEL_PARAMS = "vae_model_params"
    MSG_ARG_KEY_LSTM_MODEL_PARAMS = "lstm_model_params"

    def __init__(self, type=0, sender_id=0, receiver_id=0):
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = {}
        self.msg_params[Message.MSG_ARG_KEY_TYPE] = type
        self.msg_params[Message.MSG_ARG_KEY_SENDER] = sender_id
        self.msg_params[Message.MSG_ARG_KEY_RECEIVER] = receiver_id

    def init(self, msg_params):
        self.msg_params = msg_params

    def init_from_json_string(self, json_string):
        self.msg_params = json.loads(json_string)
        self.type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]
        # print("msg_params = " + str(self.msg_params))

    def init_from_byte_array(self, payload):
        self.msg_params = pickle.loads(payload)
        self.type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]

    def get_sender_id(self):
        return self.sender_id

    def get_receiver_id(self):
        return self.receiver_id

    def add_params(self, key, value):
        self.msg_params[key] = value

    def get_params(self):
        return self.msg_params

    def add(self, key, value):
        self.msg_params[key] = value

    def get(self, key):
        return self.msg_params[key]

    def get_type(self):
        return self.msg_params[Message.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self.msg_params

    def to_json(self):
        json_string = json.dumps(self.msg_params)
        logging.info("json string size = " + str(sys.getsizeof(json_string)))
        return json_string

    def to_byte_array(self):
        byteArr = pickle.dumps(self.msg_params)
        logging.info("byte array size = " + str(sys.getsizeof(byteArr)))
        return byteArr

    def get_content(self):
        print_dict = self.msg_params.copy()
        msg_str = str(self.__to_msg_type_string()) + ": " + str(print_dict)
        return msg_str

    def __to_msg_type_string(self):
        type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        return type
