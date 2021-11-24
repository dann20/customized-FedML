import time
import logging

import pandas as pd

import torch
from torch import nn
from torch import optim

from FedML.fedml_core.trainer.model_trainer import ModelTrainer

class AutoencoderTrainer(ModelTrainer):
    def __init__(self, autoencoder_model, train_data, device, config):
        self.id = 0
        self.model = autoencoder_model
        self.train_data = train_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.epoch_loss = list()
        self.best_model = None

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train_epoch(self, criterion, opt, epoch):
        batch_loss = list()
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
            batch_loss.append(loss.item())

        if len(batch_loss) > 0:
            self.epoch_loss.append(sum(batch_loss)/len(batch_loss))
            logging.info('Trainer_ID {}. Local Training Epoch: {} \tTotal Loss: {:.6f}'.format(self.id,
                                                                                               epoch,
                                                                                               self.epoch_loss[-1]))

        if self.epoch_loss[-1] < self.min_loss:
            torch.save(self.model.state_dict(),
                       self.config["checkpoint_dir"] + f"best_autoencoder_{epoch}.pt")
            torch.save(opt.state_dict(),
                       self.config["checkpoint_dir"] + f"optimizer_autoencoder_{epoch}.pt")
            self.min_loss = self.epoch_loss[-1]
            self.best_model = f"best_autoencoder_{epoch}.pt"

    def train(self):
        self.model.train()
        self.model.to(self.device)
        start = time.time()
        logging.info("-----START TRAINING THE AUTOENCODER-----")
        model_opt = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(self.config["auto_num_epoch"]):
            logging.info(f"Training local epoch {epoch}...")
            self.train_epoch(criterion,
                             model_opt,
                             epoch)
            logging.info(f"Done local epoch {epoch}.")
        logging.info("-----COMPLETED TRAINING THE AUTOENCODER-----")
        self.config["best_auto_model"] = self.best_model
        self.config["auto_train_time"] = (time.time() - start) / 60
        self.save_loss()

    def test(self, test_data, device, args):
        pass

    def save_loss(self):
        df_loss = pd.DataFrame([[i+1, self.epoch_loss[i]] for i in range(len(self.epoch_loss))])
        df_loss.to_csv(self.config["result_dir"] + 'autoencoder_epoch_loss.csv',
                       index=False,
                       header=['Epoch', 'Loss'])

    def get_updated_config(self):
        return self.config
