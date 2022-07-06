import sys
import logging
from copy import deepcopy

import torch

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class FedAVGTransformerTrainer(ModelTrainer):
    def __init__(self, id, transformer_model, train_data, criterion, optimizer, device, config):
        self.id = id    # id = 0 denotes server, denotes clients otherwise
        self.model = transformer_model
        self.train_data = train_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.train_loss_list = list()
        self.best_model = None
        self.criterion = criterion
        self.optimizer = optimizer

    def get_model_params(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()} if self.device is not None else self.model.state_dict()

    def set_model_params(self, model_parameters):
        if self.device is not None:
            model_parameters = {k: v.to(self.device) for k, v in model_parameters.items()}
        self.model.load_state_dict(model_parameters)

    def train_epoch(self, epoch):
        train_loss = 0.0
        self.model.train()
        for x, rul in self.train_data:
            self.model.zero_grad()
            out = self.model(x.to(self.device).float())
            loss = torch.sqrt(self.criterion(out.float(), rul.to(self.device).float()))
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(self.train_data)
        self.train_loss_list.append(train_loss)

        if train_loss < self.min_loss:
            self.min_loss = train_loss
            self.best_model = deepcopy(self.model.state_dict())
            self.best_optimizer = deepcopy(self.optimizer.state_dict())
            self.best_epoch_in_round = epoch

    def train(self):
        self.model.to(self.device)

        for epoch in range(1, self.config['n_epochs'] + 1):
            self.train_epoch(epoch)

        self.config['train_loss_list'] = self.train_loss_list

        torch.save(self.best_model, self.config["checkpoint_dir"] + "model__lr_{}_l_win_{}_dff_{}.pt".format(
                   self.config['lr'], self.config['l_win'], self.config['dff']))

    def test(self, test_data, device, args):
        ...

    def get_updated_config(self):
        return self.config

    def get_len_data(self):
        return len(self.train_data.dataset)

    def save_aggregated_model(self, round_idx):
        if self.id == 0:
            directory = self.config["server_model_dir"]
            self.config["last_aggregated_server_model"] = f"aggregated_transformer_r{round_idx}.pt"
            torch.save(self.model.state_dict(), directory + self.config["last_aggregated_server_model"])
        else:
            logging.error('Method save_aggregated_model() is supposed to be used on server after FedAvg aggregation')
            sys.exit(1)
