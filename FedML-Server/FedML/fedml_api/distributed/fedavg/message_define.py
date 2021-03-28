class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_VAE_INIT_CONFIG = 5
    MSG_TYPE_S2C_LSTM_INIT_CONFIG = 6
    MSG_TYPE_S2C_SYNC_VAE_MODEL_TO_CLIENT = 7
    MSG_TYPE_S2C_SYNC_LSTM_MODEL_TO_CLIENT = 8

    # client to server
    MSG_TYPE_C2S_SEND_VAE_MODEL_TO_SERVER = 9
    MSG_TYPE_C2S_SEND_LSTM_MODEL_TO_SERVER = 10

    MSG_TYPE_C2S_ACTIVATE_LSTM_PHASE = 11

    # original
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_VAE_MODEL_PARAMS = "vae_model_params"
    MSG_ARG_KEY_LSTM_MODEL_PARAMS = "lstm_model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"


