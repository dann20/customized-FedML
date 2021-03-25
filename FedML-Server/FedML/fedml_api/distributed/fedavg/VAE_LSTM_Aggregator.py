import os
import sys
import tensorflow as tf
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from FedML.fedml_api.data_preprocessing.VAE_LSTM.data_loader import DataGenerator
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Models import VAEmodel, lstmKerasModel
from FedML.fedml_api.distributed.fedavg.VAE_LSTM_Trainer import vaeTrainer
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import process_config, create_dirs, get_args, save_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Aggregator:
    def __init__(self, vae_trainers, global_vae_trainer, lstm_models, global_lstm_model, config, weights):
        self.global_lstm_model = global_lstm_model
        self.vae_trainers = vae_trainers
        self.global_vae_trainer = global_vae_trainer
        self.lstm_models = lstm_models
        self.config = config
        self.num_comm_rounds = self.config["num_comm_rounds"]
        self.weights = weights

    def run(self):
        self.aggregate_vae()
        self.aggregate_lstm()

    def aggregate_vae(self):
        if self.config['TRAIN_VAE']:
            if self.config['vae_epochs_per_comm_round'] > 0:
                for comm_round in range(self.config['num_comm_rounds']):
                    self.train_vars_VAE_of_clients = []
                    for vae_trainer in self.vae_trainers:
                        for i in range(len(vae_trainer.model.train_vars_VAE)):
                            vae_trainer.model.train_vars_VAE[i].load(self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess), vae_trainer.sess)
                            # print(np.multiply(vae_trainer.model.train_vars_VAE[i].eval(vae_trainer.sess), self.weights[0]))
                        vae_trainer.train()
                        self.train_vars_VAE_of_clients.append(vae_trainer.model.train_vars_VAE)
                    for i in range(len(self.vae_trainers[0].model.train_vars_VAE)):
                        global_train_var_eval = np.zeros_like(self.vae_trainers[0].model.train_vars_VAE[i].eval(self.vae_trainers[0].sess))
                        for j in range(len(self.vae_trainers)):
                            global_train_var_eval += np.multiply(self.weights[j], self.vae_trainers[j].model.train_vars_VAE[i].eval(self.vae_trainers[j].sess))
                        self.global_vae_trainer.model.train_vars_VAE[i].load(global_train_var_eval, self.global_vae_trainer.sess)
                        # print(self.global_vae_trainer.model.train_vars_VAE[i].eval(self.global_vae_trainer.sess))
                #save model global
                self.global_vae_trainer.model.save(self.global_vae_trainer.sess)

    def aggregate_lstm(self):
        if self.config['TRAIN_LSTM']:
            if self.config['lstm_epochs_per_comm_round'] > 0:
                for comm_round in range(self.config['num_comm_rounds']):
                    lstm_weights = []
                    global_weights = self.global_lstm_model.lstm_nn_model.get_weights()
                    for i in range(len(self.lstm_models)):
                        lstm_model = self.lstm_models[i]

                        # produce the embedding of all sequences for training of lstm model
                        # process the windows in sequence to get their VAE embeddings
                        # lstm_model.produce_embeddings(self.vae_trainers[i].model, self.vae_trainers[i].data, self.vae_trainers[i].sess)
                        lstm_model.produce_embeddings(self.global_vae_trainer.model, self.global_vae_trainer.data, self.global_vae_trainer.sess)

                        # Create a basic model instance
                        # lstm_nn_model = lstm_model.create_lstm_model(self.config)
                        lstm_nn_model = lstm_model.lstm_nn_model
                        lstm_nn_model.set_weights(global_weights)
                        lstm_nn_model.summary()   # Display the model's architecture
                        # checkpoint path
                        checkpoint_path = self.config['checkpoint_dir_lstm']\
                                          + "cp_{}.ckpt".format(lstm_model.name)
                        # Create a callback that saves the model's weights
                        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                          save_weights_only=True,
                                                                          verbose=1)
                        # load weights if possible
                        # lstm_model.load_model(lstm_nn_model, checkpoint_path)

                        # start training
                        if self.config['lstm_epochs_per_comm_round'] > 0:
                            lstm_model.train(lstm_nn_model, cp_callback)
                        lstm_weights.append(lstm_nn_model.get_weights())
                        # set globel_weights = 0
                        # global_weights = np.subtract(global_weights, global_weights)
                    for i in range(len(self.lstm_models)):
                        if i == 0:
                            global_weights = np.multiply(lstm_weights[i], self.weights[i])
                        else:
                            global_weights += np.multiply(lstm_weights[i], self.weights[i])
                        self.global_lstm_model.lstm_nn_model.set_weights(global_weights)
                        # make a prediction on the test set using the trained model
                        # lstm_embedding = lstm_nn_model.predict(lstm_model.x_test, batch_size=self.config['batch_size_lstm'])
                        # print(lstm_embedding.shape)
                                    # save global lstm model
                glb_checkpoint_path = self.config['checkpoint_dir_lstm'] + "cp_{}.ckpt".format(self.global_lstm_model.name)
                self.global_lstm_model.lstm_nn_model.save_weights(glb_checkpoint_path)
