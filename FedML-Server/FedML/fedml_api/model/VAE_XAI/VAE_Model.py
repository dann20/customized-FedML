import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sampling_z(args):
  """Reparameterization trick by sampling from an isotropic unit Gaussian.

  # Arguments
      args (tensor): mean and log of variance of Q(z|X)

  # Returns
      z (tensor): sampled latent vector
  """

  z_mean, z_log_var = args
  batch = K.shape(z_mean)[0]
  dim = K.int_shape(z_mean)[1]

  # by default, random_normal has mean = 0 and std = 1.0
  epsilon = K.random_normal(shape=(batch, dim))
  return z_mean + K.exp(0.5 * z_log_var) * epsilon      # basically creating the distribution of z, Q(z|X)

class VAEmodel:
    def __init__(self, config, name):
        self.name = name
        self.add_config(config)
        self.model = self.create_vae_model()

    def add_config(self, config):
        self.original_dim = config["original_dim"]
        self.input_shape = (self.original_dim,)
        self.intermediate_dim = config["intermediate_dim"]
        self.batch_size = config["batch_size"]
        self.latent_dim = config["latent_dim"]
        self.epochs = config["epochs"]
        self.lr = config["learning_rate"]
        self.result_dir = config["result_dir"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.checkpoint_path = self.checkpoint_dir\
                                          + "cp_{}.ckpt".format(self.name)

    def create_vae_model(self):
        # build encoder model: Q(z|X)
        inputs = tf.keras.Input(shape=self.input_shape, name='encoder_input')
        x = tf.keras.layers.Dense(self.intermediate_dim[0], activation='tanh')(inputs)
        x = tf.keras.layers.Dense(self.intermediate_dim[1], activation='relu')(x)
        # x = tf.keras.layers.Dense(intermediate_dim[2], activation='relu')(x)

        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that 'output_shape' isn't necessary with the TensorFlow backend
        z = tf.keras.layers.Lambda(sampling_z, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])  # takes in z_mean, z_log_var and return z, using function sampling_z

        # instantiate encoder model
        encoder = tf.keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        #plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

        # build decoder model: P(X|z)
        latent_inputs = tf.keras.Input(shape=(self.latent_dim, ), name='z_sampling')
        x = tf.keras.layers.Dense(self.intermediate_dim[1], activation='tanh')(latent_inputs)
        x = tf.keras.layers.Dense(self.intermediate_dim[0], activation='relu')(x)
        # x = tf.keras.layers.Dense(intermediate_dim[0], activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.original_dim, activation='relu')(x)

        # instantiate decoder model
        decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name='decoder')
        decoder.summary()
        #plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.Model(inputs, outputs, name=self.name)
        vae.summary()
        #plot_model(vae, to_file='vae_model.png', show_shapes=True)
        reconstruction_loss = mse(inputs, outputs)
        # reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.original_dim
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        # kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # E[log P(X|z)]
        # reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=1)
        vae_loss = 0.5*reconstruction_loss + 0.5*kl_loss
        vae.add_loss(vae_loss)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        vae.compile(optimizer=optimizer)
        return vae

    def get_vae_model_params(self):
        model_params = self.model.get_weights()
        # for i in range(len(model_params)):
        #     model_params[i] = model_params[i].astype(np.float16, copy=False)
        return model_params

    def set_vae_model_params(self, weights):
        self.model.set_weights(weights)

    def load_train_data(self, train_data, val_data):
        self.normal_train_data = train_data
        self.normal_val_data = val_data

    def load_test_data(self, test_data, test_labels):
        self.test_data = test_data
        self.test_labels = test_labels

    def train(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                          save_weights_only=True,
                                                          verbose=1)

        self.history = self.model.fit(
            self.normal_train_data, self.normal_train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.normal_val_data, self.normal_val_data),
            callbacks=[cp_callback],
            verbose=1
        )
        self.plot_train_and_val_loss()

    def plot_train_and_val_loss(self):
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(self.result_dir + '/train-val-loss-{}.png'.format(self.name))

    def plot_test_loss(self, loss):
        plt.hist(loss[None,:], bins=50)
        plt.xlabel("Loss")
        plt.ylabel("No of samples")
        plt.savefig(self.result_dir + '/test-loss-{}.png'.format(self.name))

    def set_threshold(self, threshold):
        self.threshold = threshold

    def test(self):
        reconstructions = self.model.predict(self.test_data)
        error_vector = np.subtract(reconstructions, self.test_data)
        error_vector = np.concatenate([error_vector, self.test_labels.to_numpy().reshape(-1, 1)], axis=1) # pandas dataframe to numpy arr
        np.savetxt(self.result_dir + 'error_vector.csv', error_vector, delimiter=',')
        loss = mse(self.test_data, reconstructions)
        self.plot_test_loss(loss)
        try:
            preds = tf.math.less(loss, self.threshold)
        except:
            print('Not set threshold.')
        print(preds)
        print("Accuracy = {}".format(accuracy_score(self.test_labels, preds)))
        print("Precision = {}".format(precision_score(self.test_labels, preds)))
        print("Recall = {}".format(recall_score(self.test_labels, preds)))
        print("F1_Score = {}".format(f1_score(self.test_labels, preds)))

    def save_model(self):
        self.model.save_weights(self.checkpoint_dir)

    def load_model(self):
        self.model.load_weights(self.checkpoint_path)
