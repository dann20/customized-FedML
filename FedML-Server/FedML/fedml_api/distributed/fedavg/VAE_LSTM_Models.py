import os
import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from FedML.fedml_core.trainer.base_vae_lstm import BaseModel

tfd = tfp.distributions

class VAEmodel(BaseModel):
  def __init__(self, config, name):
    super(VAEmodel, self).__init__(config, name)
    self.input_dims = self.config['l_win'] * self.config['n_channel']

    self.define_iterator()
    self.build_model()
    self.define_loss()
    self.training_variables()
    self.compute_gradients()
    self.init_saver()

  def define_iterator(self):
    self.original_signal = tf.compat.v1.placeholder(tf.float32, [None, self.config['l_win'], self.config['n_channel']])
    self.seed = tf.compat.v1.placeholder(tf.int64, shape=())
    self.dataset = tf.data.Dataset.from_tensor_slices(self.original_signal)
    self.dataset = self.dataset.shuffle(buffer_size=60000, seed=self.seed)
    self.dataset = self.dataset.repeat(8000)
    self.dataset = self.dataset.batch(self.config['batch_size'], drop_remainder=True)
    self.iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
    self.input_image = self.iterator.get_next()
    self.code_input = tf.compat.v1.placeholder(tf.float32, [None, self.config['code_size']])
    self.is_code_input = tf.compat.v1.placeholder(tf.bool)
    self.sigma2_offset = tf.constant(self.config['sigma2_offset'])

  def build_model(self):
    init = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    with tf.compat.v1.variable_scope('encoder_{}'.format(self.name)):
      #if self.config['n_channel']==1: #them doan nay
      input_tensor = tf.expand_dims(self.original_signal, -1)
      #    print(input_tensor.shape)
      #else:
      #    input_tensor = self.original_signal
      #if self.config['l_win'] == 24:
      conv_1 = tf.compat.v1.layers.dense(inputs=input_tensor,
                                  units = self.config['n_channel']*10,
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
      print("conv_1: {}".format(conv_1))


      encoded_signal = tf.compat.v1.layers.flatten(conv_1)
      encoded_signal = tf.compat.v1.layers.dense(encoded_signal,
                                       units=self.config['code_size'] * 4,
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=init)
      self.code_mean = tf.compat.v1.layers.dense(encoded_signal,
                                       units=self.config['code_size'],
                                       activation=None,
                                       kernel_initializer=init,
                                       name='code_mean')
      self.code_std_dev = tf.compat.v1.layers.dense(encoded_signal,
                                          units=self.config['code_size'],
                                          activation=tf.nn.relu,
                                          kernel_initializer=init,
                                          name='code_std_dev')
      self.code_std_dev = self.code_std_dev + 1e-2
      #mvn = tfp.distributions.MultivariateNormalDiag(loc=self.code_mean, scale_diag=self.code_std_dev)
      #self.code_sample = mvn.sample()
      mvn = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros([self.config['code_size']]), scale_diag=tf.ones([self.config['code_size']]))
      self.code_sample = tf.add(self.code_mean , self.code_std_dev * mvn.sample())
    print("finish encoder: \n{}".format(self.code_sample))
    print("\n")

    with tf.compat.v1.variable_scope('decoder_{}'.format(self.name)):
      encoded = tf.cond(pred=self.is_code_input, true_fn=lambda: self.code_input, false_fn=lambda: self.code_sample)
      decoded_1 = tf.compat.v1.layers.dense(encoded,
                                  units=self.config['code_size'] * 4,
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
      #decoded_1 = tf.reshape(decoded_1, [-1, 1, 1, self.config['num_hidden_units']])
      if self.config['l_win'] == 24:
        decoded_2 = tf.compat.v1.layers.dense(decoded_1,
                                  units=self.config['n_channel']*10,
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=init)
        decoded = tf.compat.v1.layers.dense(decoded_2,
                                   units=self.config['n_channel']*self.config['l_win'],
                                   activation=None,
                                   kernel_initializer=init)
        print("decoded is: {}".format(decoded))
        self.decoded = tf.reshape(decoded, [-1, self.config['l_win'], self.config['n_channel']])
    print("finish decoder: \n{}".format(self.decoded))
    print('\n')

    # define sigma2 parameter to be trained to optimise ELBO
    with tf.compat.v1.variable_scope('sigma2_dataset_{}'.format(self.name)):
      if self.config['TRAIN_sigma'] == 1:
        sigma = tf.Variable(tf.cast(self.config['sigma'], tf.float32),
                            dtype=tf.float32, trainable=True)
      else:
        sigma = tf.cast(self.config['sigma'], tf.float32)
      self.sigma2 = tf.square(sigma)
      if self.config['TRAIN_sigma'] == 1:
        self.sigma2 = self.sigma2 + self.sigma2_offset

    print("sigma2: \n{}\n".format(self.sigma2))

# This LSTM Keras Model also acts as a LSTM Trainer
class lstmKerasModel:
    def __init__(self, name, config, embeddings=None):
        self.name = name
        self.config = config
        self.lstm_nn_model = self.create_lstm_model()
        if embeddings:
            self.x_train = embeddings.x_train
            self.y_train = embeddings.y_train
            self.x_test = embeddings.x_test
            self.y_test = embeddings.y_test
            self.embedding_lstm_train = embeddings.embedding_lstm_train
            self.embedding_lstm_test = embeddings.embedding_lstm_test

    def set_embeddings(self, embeddings):
        self.x_train = embeddings.x_train
        self.y_train = embeddings.y_train
        self.x_test = embeddings.x_test
        self.y_test = embeddings.y_test
        self.embedding_lstm_train = embeddings.embedding_lstm_train
        self.embedding_lstm_test = embeddings.embedding_lstm_test

    def create_lstm_model(self):
        config = self.config
        lstm_input = tf.keras.layers.Input(shape=(config['l_seq'] - 1, config['code_size']))
        print ("lstm input:", lstm_input.shape)
        LSTM1 = tf.keras.layers.LSTM(config['num_hidden_units_lstm'], return_sequences=True)(lstm_input)
        LSTM2 = tf.keras.layers.LSTM(config['num_hidden_units_lstm'], return_sequences=True)(LSTM1)
        lstm_output = tf.keras.layers.LSTM(config['code_size'], return_sequences=True, activation=None)(LSTM2)
        lstm_model = tf.keras.Model(lstm_input, lstm_output)
        lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_lstm']),
                           loss='mse',
                           metrics=['mse'])
        return lstm_model

    def produce_embeddings(self, model_vae, data, sess):
        config = self.config
        self.embedding_lstm_train = np.zeros((data.n_train_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_train_lstm):
            feed_dict = {model_vae.original_signal: np.reshape(data.train_set_lstm['data'][i],(-1,config['l_win'],config['n_channel'])), #sua dong nay
                         model_vae.is_code_input: False,
                         model_vae.code_input: np.zeros((1, config['code_size']))}
            self.embedding_lstm_train[i] = sess.run(model_vae.code_mean, feed_dict=feed_dict)
        print("Finish processing the embeddings of the entire dataset of {}.".format(model_vae.name))
        #print("The first a few embeddings are\n{}".format(self.embedding_lstm_train[0, 0:5]))
        self.x_train = self.embedding_lstm_train[:, :config['l_seq'] - 1]
        self.y_train = self.embedding_lstm_train[:, 1:]

        self.embedding_lstm_test = np.zeros((data.n_val_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_val_lstm):
            feed_dict = {model_vae.original_signal: np.reshape(data.val_set_lstm['data'][i], (-1, config['l_win'], config['n_channel'])),
                        model_vae.is_code_input: False,
                        model_vae.code_input: np.zeros((1, config['code_size']))}
            self.embedding_lstm_test[i] = sess.run(model_vae.code_mean, feed_dict=feed_dict)
        self.x_test = self.embedding_lstm_test[:, :config['l_seq'] - 1]
        self.y_test = self.embedding_lstm_test[:, 1:]

    def load_model(self, lstm_model, checkpoint_path):
        config = self.config
        print(config['checkpoint_dir_lstm'] + 'checkpoint')
        if os.path.isfile(config['checkpoint_dir_lstm'] + 'checkpoint'):#"_{}".format(self.name) + '/checkpoint'):
            try:
                lstm_model.load_weights(checkpoint_path)
                print("LSTM model loaded.")
            except:
                print("No LSTM model loaded.")
        else:
            print("No LSTM model loaded.")

    def train(self, lstm_model, cp_callback):
        print('checkpoint 1')
        config = self.config
        print('checkpoint 2')
        lstm_model.fit(self.x_train, self.y_train,
                       validation_data=(self.x_test, self.y_test),
                       batch_size=config['batch_size_lstm'],
                       epochs=config['lstm_epochs_per_comm_round'],
                       callbacks=[cp_callback],
                       verbose=1)
        print('checkpoint 3')

    def plot_reconstructed_lt_seq(self, idx_test, model_vae, sess, data, lstm_embedding_test):
        config = self.config
        feed_dict_vae = {model_vae.original_signal: np.zeros((config['l_seq'], config['l_win'], config['n_channel'])),
                         model_vae.is_code_input: True,
                         model_vae.code_input: self.embedding_lstm_test[idx_test]}
        decoded_seq_vae = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_vae))
        print("Decoded seq from VAE: {}".format(decoded_seq_vae.shape))

        feed_dict_lstm = {model_vae.original_signal: np.zeros((config['l_seq'] - 1, config['l_win'], config['n_channel'])),
                          model_vae.is_code_input: True,
                          model_vae.code_input: lstm_embedding_test[idx_test]}
        decoded_seq_lstm = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_lstm))
        print("Decoded seq from lstm: {}".format(decoded_seq_lstm.shape))

        fig, axs = plt.subplots(config['n_channel'], 2, figsize=(15, 4.5 * config['n_channel']), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()
        for j in range(config['n_channel']):
            for i in range(2):
                axs[i + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                                    np.reshape(data.val_set_lstm['data'][idx_test, :, :, j],
                                               (config['l_seq'] * config['l_win'])))
                axs[i + j * 2].grid(True)
                axs[i + j * 2].set_xlim(0, config['l_seq'] * config['l_win'])
                axs[i + j * 2].set_xlabel('samples')
            if config['n_channel'] == 1:
                axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_vae, (config['l_seq'] * config['l_win'])), 'r--')
                axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_lstm, ((config['l_seq'] - 1) * config['l_win'])), 'g--')
            else:
                axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_vae[:, :, j], (config['l_seq'] * config['l_win'])), 'r--')
                axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
                                    np.reshape(decoded_seq_lstm[:, :, j], ((config['l_seq'] - 1) * config['l_win'])), 'g--')
            axs[0 + j * 2].set_title('VAE reconstruction - channel {}'.format(j))
            axs[1 + j * 2].set_title('LSTM reconstruction - channel {}'.format(j))
            for i in range(2):
                axs[i + j * 2].legend(('ground truth', 'reconstruction'))
            savefig(config['result_dir'] + "_{}".format(self.name) + "/lstm_long_seq_recons_{}.pdf".format(idx_test))
            fig.clf()
            plt.close()

    def plot_lstm_embedding_prediction(self, idx_test, model_vae, sess, data, lstm_embedding_test):
        config = self.config
        self.plot_reconstructed_lt_seq(idx_test, config, model_vae, sess, data, lstm_embedding_test)

        fig, axs = plt.subplots(2, config['code_size'] // 2, figsize=(15, 5.5), edgecolor='k')
        fig.subplots_adjust(hspace=.4, wspace=.4)
        axs = axs.ravel()
        for i in range(config['code_size']):
            axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(self.embedding_lstm_test[idx_test, 1:, i]))
            axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(lstm_embedding_test[idx_test, :, i]))
            axs[i].set_xlim(1, config['l_seq'] - 1)
            axs[i].set_ylim(-2.5, 2.5)
            axs[i].grid(True)
            axs[i].set_title('Embedding dim {}'.format(i))
            axs[i].set_xlabel('windows')
            if i == config['code_size'] - 1:
                axs[i].legend(('VAE\nembedding', 'LSTM\nembedding'))
        savefig(config['result_dir'] + "_{}".format(self.name) + "lstm_seq_embedding_{}.pdf".format(idx_test))
        fig.clf()
        plt.close()

    def get_lstm_model_params(self):
        try:
            global_model_params = self.lstm_nn_model.get_weights()
        except:
            print('Function get_lstm_model_params failed')
        else:
            for i in range(len(global_model_params)):
                global_model_params[i] = global_model_params[i].astype(np.float16, copy=False)
            return global_model_params # returns list of arrays

    def set_lstm_model_params(self, weights): # args is list of arrays
        # print(weights)
        try:
            self.lstm_nn_model.set_weights(weights)
        except:
            print('Funtion set_lstm_model_params failed')
        self.lstm_nn_model.summary()

class ProduceEmbeddings:
    def __init__(self, model_vae, data, sess, config):
        self.embedding_lstm_train = np.zeros((data.n_train_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_train_lstm):
            feed_dict = {model_vae.original_signal: np.reshape(data.train_set_lstm['data'][i],(-1,config['l_win'],config['n_channel'])), #sua dong nay
                         model_vae.is_code_input: False,
                         model_vae.code_input: np.zeros((1, config['code_size']))}
            self.embedding_lstm_train[i] = sess.run(model_vae.code_mean, feed_dict=feed_dict)
        print("Finish processing the embeddings of the entire dataset of {}.".format(model_vae.name))
        #print("The first a few embeddings are\n{}".format(self.embedding_lstm_train[0, 0:5]))
        self.x_train = self.embedding_lstm_train[:, :config['l_seq'] - 1]
        self.y_train = self.embedding_lstm_train[:, 1:]

        self.embedding_lstm_test = np.zeros((data.n_val_lstm, config['l_seq'], config['code_size']))
        for i in range(data.n_val_lstm):
            feed_dict = {model_vae.original_signal: np.reshape(data.val_set_lstm['data'][i], (-1, config['l_win'], config['n_channel'])),
                        model_vae.is_code_input: False,
                        model_vae.code_input: np.zeros((1, config['code_size']))}
            self.embedding_lstm_test[i] = sess.run(model_vae.code_mean, feed_dict=feed_dict)
        self.x_test = self.embedding_lstm_test[:, :config['l_seq'] - 1]
        self.y_test = self.embedding_lstm_test[:, 1:]
