import tensorflow as tf
import os
import tensorflow_probability as tfp
import random
import numpy as np
import time
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, savefig, figure
from FedML.fedml_api.distributed.fedavg.utils_VAE_LSTM import count_trainable_variables
tfd = tfp.distributions


class BaseDataGenerator:

    def __init__(self, config):
        self.config = config

    # separate training and val sets
    def separate_train_and_val_set(self, n_win):
        n_train = int(np.floor((n_win * 0.9)))
        n_val = n_win - n_train
        idx_train = random.sample(range(n_win), n_train)
        idx_val = list(set(idx_train) ^ set(range(n_win)))
        return idx_train, idx_val, n_train, n_val


class BaseModel:

    def __init__(self, config, name):
        self.name = name
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.two_pi = tf.constant(2 * np.pi)


    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, save_path=self.config['checkpoint_dir'] + self.name + "/",
                        global_step=self.global_step_tensor)
        print("Model saved.")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        print("checkpoint_dir at loading: {}".format(self.config['checkpoint_dir'] + self.name))
        latest_checkpoint = tf.train.latest_checkpoint(self.config['checkpoint_dir'] + self.name)

        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded.")
        else:
            print("No model loaded.")

    # initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.compat.v1.variable_scope('cur_epoch_{}'.format(self.name)):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.compat.v1.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.compat.v1.variable_scope('global_step_{}'.format(self.name)):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.compat.v1.assign(
                self.global_step_tensor, self.global_step_tensor + 1)

    def define_loss(self):
        with tf.compat.v1.name_scope("loss_{}".format(self.name)):
            # KL divergence loss - analytical result
            KL_loss = 0.5 * (tf.reduce_sum(input_tensor=tf.square(self.code_mean), axis=1)
                             + tf.reduce_sum(input_tensor=tf.square(self.code_std_dev), axis=1)
                             - tf.reduce_sum(input_tensor=tf.math.log(tf.square(self.code_std_dev)), axis=1)
                             - self.config['code_size'])
            #with tf.Session() as sess:
            #    print ("KL_loss shape1", sess.run(tf.shape(KL_loss)))
            self.KL_loss = tf.reduce_mean(input_tensor=KL_loss)
            print ("KL_loss shape2", self.KL_loss.get_shape())

            # norm 1 of standard deviation of the sample-wise encoder prediction
            self.std_dev_norm = tf.reduce_mean(input_tensor=self.code_std_dev, axis=0)

            weighted_reconstruction_error_dataset = tf.reduce_sum(
                input_tensor=tf.square(self.original_signal - self.decoded), axis=[1, 2])
            weighted_reconstruction_error_dataset = tf.reduce_mean(input_tensor=weighted_reconstruction_error_dataset)
            self.weighted_reconstruction_error_dataset = weighted_reconstruction_error_dataset / (2 * self.sigma2)
            print ("recon_error shape", self.weighted_reconstruction_error_dataset.get_shape())

            # least squared reconstruction error
            ls_reconstruction_error = tf.reduce_sum(
                input_tensor=tf.square(self.original_signal - self.decoded), axis=[1, 2])
            self.ls_reconstruction_error = tf.reduce_mean(input_tensor=ls_reconstruction_error)
            print ("ls_recon shape", self.ls_reconstruction_error.get_shape())

            # sigma regularisor - input elbo
            self.sigma_regularisor_dataset = self.input_dims / 2 * tf.math.log(self.sigma2)
            print ("sigma_regu shape", self.sigma_regularisor_dataset.shape)
            two_pi = self.input_dims / 2 * tf.constant(2 * np.pi)
            print ("two_pi shape", two_pi.shape)

            self.elbo_loss = two_pi + self.sigma_regularisor_dataset + \
                             0.5 * self.weighted_reconstruction_error_dataset + self.KL_loss
            with tf.compat.v1.Session() as sess:
                print ("elbo loss shape", sess.run(tf.shape(input=self.elbo_loss)))

    def training_variables(self):
        encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "encoder_{}".format(self.name))
        decoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "decoder_{}".format(self.name))
        sigma_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "sigma2_dataset_{}".format(self.name))
        self.train_vars_VAE = encoder_vars + decoder_vars + sigma_vars

        num_encoder = count_trainable_variables('encoder_{}'.format(self.name))
        num_decoder = count_trainable_variables('decoder_{}'.format(self.name))
        num_sigma2 = count_trainable_variables('sigma2_dataset_{}'.format(self.name))
        self.num_vars_total = num_decoder + num_encoder + num_sigma2
        print("Total number of trainable parameters in the VAE network of {} is: {}".format(self.name, self.num_vars_total))

    def compute_gradients(self):
        self.lr = tf.compat.v1.placeholder(tf.float32, [])
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.95)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        gvs_dataset = opt.compute_gradients(self.elbo_loss, var_list=self.train_vars_VAE)
        print('gvs for dataset: {}'.format(gvs_dataset))
        capped_gvs = [(self.ClipIfNotNone(grad), var) for grad, var in gvs_dataset]

        with tf.control_dependencies(update_ops):
            self.train_step_gradient = opt.apply_gradients(capped_gvs)
        print("Reach the definition of loss for VAE")

    def ClipIfNotNone(self, grad):
        if grad is None:
            return grad
        return tf.clip_by_value(grad, -1, 1)

    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1, var_list=self.train_vars_VAE)


class BaseTrain:
    def __init__(self, sess, model, data, config):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.compat.v1.global_variables_initializer(),
                             tf.compat.v1.local_variables_initializer())
        self.sess.run(self.init)

        # keep a record of the training result
        self.train_loss = []
        self.val_loss = []
        self.train_loss_ave_epoch = []
        self.val_loss_ave_epoch = []
        self.recons_loss_train = []
        self.recons_loss_val = []
        self.KL_loss_train = []
        self.KL_loss_val = []
        self.sample_std_dev_train = []
        self.sample_std_dev_val = []
        self.iter_epochs_list = []
        self.test_sigma2 = []

    def train(self):
        self.start_time = time.time()
        for cur_epoch in range(0, self.config['vae_epochs_per_comm_round'], 1):
            self.train_epoch()

        # compute current execution time
        # self.current_time = time.time()
        # elapsed_time = (self.current_time - self.start_time) / 60
        # est_remaining_time = (
        #                             self.current_time - self.start_time) / (cur_epoch + 1) * (
        #                              self.config['num_comm_rounds'] - cur_epoch - 1)
        # est_remaining_time = est_remaining_time / 60
        # print("Already trained for {} min; Remaining {} min.".format(elapsed_time, est_remaining_time))
        self.sess.run(self.model.increment_cur_epoch_tensor)

    def save_variables_VAE(self):
        # save some variables for later inspection
        file_name = "{}{}-batch-{}-round-{}-code-{}-lr-{}-model-{}.npz".format(self.config['result_dir'],
                                                                      self.config['exp_name'],
                                                                      self.config['batch_size'],
                                                                      self.config['num_comm_rounds'],
                                                                      self.config['code_size'],
                                                                      self.config['learning_rate_vae'],
                                                                      self.model.name)
        np.savez(file_name,
                 iter_list_val=self.iter_epochs_list,
                 train_loss=self.train_loss,
                 val_loss=self.val_loss,
                 n_train_iter=self.n_train_iter,
                 n_val_iter=self.n_val_iter,
                 recons_loss_train=self.recons_loss_train,
                 recons_loss_val=self.recons_loss_val,
                 KL_loss_train=self.KL_loss_train,
                 KL_loss_val=self.KL_loss_val,
                 num_para_all=self.model.num_vars_total,
                 sigma2=self.test_sigma2)

    def plot_train_and_val_loss(self):
        # plot the training and validation loss over epochs
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plot(self.train_loss, 'b-')
        plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
        plt.legend(('training loss (total)', 'validation loss'))
        plt.title('training loss over iterations (val @ epochs)')
        plt.ylabel('total loss')
        plt.xlabel('iterations')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/loss_{}.png'.format(self.model.name))

        # plot individual components of validation loss over epochs
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plot(self.recons_loss_val, 'b-')
        plot(self.KL_loss_val, 'r-')
        plt.legend(('Reconstruction loss', 'KL loss'))
        plt.title('validation loss breakdown')
        plt.ylabel('loss')
        plt.xlabel('num of batch')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/val-loss_{}.png'.format(self.model.name))

        # plot individual components of validation loss over epochs
        plt.clf()
        figure(num=1, figsize=(8, 6))
        plot(self.test_sigma2, 'b-')
        plt.title('sigma2 over training')
        plt.ylabel('sigma2')
        plt.xlabel('iter')
        plt.grid(True)
        savefig(self.config['result_dir'] + '/sigma2_{}.png'.format(self.model.name))
