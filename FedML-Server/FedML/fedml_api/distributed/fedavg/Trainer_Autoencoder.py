import os
import time
import logging
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim

from FedML.fedml_core.trainer.model_trainer import ModelTrainer

class AutoencoderTrainer(ModelTrainer):
    def __init__(self, id, autoencoder_model, train_data, val_data, device, config):
        self.id = id
        self.model = autoencoder_model
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.train_loss_list = list()
        self.val_loss_list = list()
        self.best_model = None
        self.best_epoch = 0
        self.best_optimizer = None

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train_epoch(self, criterion, opt, epoch):
        train_loss = 0.0
        self.model.train()
        for i, batch in enumerate(self.train_data):
            src = batch["input"].float()
            src = src.to(self.device)
            trg = batch["target"].float()
            trg = trg.to(self.device)
            out = self.model(src)

            opt.zero_grad()
            assert out.size(1) == trg.size(1)
            loss = criterion(out, trg)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        train_loss = train_loss / len(self.train_data)
        self.train_loss_list.append(train_loss)

        logging.info('Trainer_ID {}. Local Training Epoch: {} \tTrain Loss: {:.6f}'.format(self.id, epoch, train_loss))

        if self.val_data is None:
            if train_loss < self.min_loss:
                self.min_loss = train_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_optimizer = deepcopy(opt.state_dict())
                self.best_epoch = epoch
        else:
            self.validate_epoch(criterion, opt, epoch)

    def validate_epoch(self, criterion, opt, epoch):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_data):
                src = batch["input"].float()
                src = src.to(self.device)
                trg = batch["target"].float()
                trg = trg.to(self.device)
                out = self.model(src)

                loss = criterion(out, trg)
                val_loss += loss.item()

        val_loss = val_loss / len(self.val_data)
        self.val_loss_list.append(val_loss)
        logging.info('Trainer_ID {}. Local Training Epoch: {} \tValidation Loss: {:.6f}'.format(self.id, epoch, val_loss))

        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.best_model = deepcopy(self.model.state_dict())
            self.best_optimizer = deepcopy(opt.state_dict())
            self.best_epoch = epoch

    def train(self):
        self.model.to(self.device)

        start = time.perf_counter()
        logging.info("-----START TRAINING THE AUTOENCODER-----")
        model_opt = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(1, self.config["auto_num_epoch"] + 1):
            logging.info(f"Training local epoch {epoch}...")
            self.train_epoch(criterion,
                             model_opt,
                             epoch)
            logging.info(f"Done local epoch {epoch}.")
        logging.info("-----COMPLETED TRAINING THE AUTOENCODER-----")
        self.config["auto_train_time"] = (time.perf_counter() - start) / 60

        torch.save(self.best_model, self.config["checkpoint_dir"] + "autoencoder_model.pt")
        torch.save(self.best_optimizer, self.config["checkpoint_dir"] + "autoencoder_opt.pt")
        self.config["best_auto_epoch"] = self.best_epoch
        self.save_loss()
        self.client_plot_loss()

    def test(self, test_data, device, args):
        pass

    def load_model(self, path = None):
        if path is None:
            path = self.config['checkpoint_dir'] + "autoencoder_model.pt"
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save_loss(self):
        if self.val_data is not None:
            df_loss = pd.DataFrame([[i+1, self.train_loss_list[i], self.val_loss_list[i]] for i in range(len(self.train_loss_list))])
            df_loss.to_csv(self.config["result_dir"] + 'autoencoder_epoch_loss.csv',
                           index=False,
                           header=['Epoch', 'TrainingLoss', 'ValidationLoss'])
        else:
            df_loss = pd.DataFrame([[i+1, self.train_loss_list[i]] for i in range(len(self.train_loss_list))])
            df_loss.to_csv(self.config["result_dir"] + 'autoencoder_epoch_loss.csv',
                           index=False,
                           header=['Epoch', 'TrainingLoss'])

    def client_plot_loss(self):
        epochs = range(1, self.config["auto_num_epoch"] + 1)
        plt.plot(epochs, self.train_loss_list, 'g', label='Training loss')
        if self.val_data is not None:
            plt.plot(epochs, self.val_loss_list, 'b', label='Validation loss')
            plt.title('{}. ID {}: Training and Validation Loss'.format("AUTOENCODER", self.id))
        else:
            plt.title('{}. ID {}: Training Loss'.format("AUTOENCODER", self.id))
        plt.xlabel('Local Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config["result_dir"], "autoencoder_loss.png"), dpi=300)
        plt.close()

    def get_updated_config(self):
        return self.config
