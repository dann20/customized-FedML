import time
import logging
import os
import sys
import math
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn

from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_api.distributed.scaffold.SCAFFOLDOptimizer import SCAFFOLDOptimizer

class SCAFFOLDTransformerTrainer(ModelTrainer):
    def __init__(self, id, autoencoder_model, transformer_model, train_data, val_data, device, config):
        self.id = id    # id = 0 denotes server, denotes clients otherwise
        self.model = transformer_model
        if autoencoder_model is not None:
            self.encoder = autoencoder_model.encoder
            self.encoder.eval()
        self.train_data = train_data
        self.val_data = val_data
        self.num_train_samples = len(self.train_data.dataset) if self.id != 0 else None
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.train_loss_list = list()
        self.val_loss_list = list()
        self.best_model = None
        self.best_epoch_in_round = 0
        self.best_optimizer = None
        self.mask = self._create_mask()

        if self.id == 0:
            self.server_controls = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]
        else:
            self.server_model = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]
            self.controls = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]
            self.server_controls = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]
            self.delta_model = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]
            self.delta_controls = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]

    def set_model_params(self, model_parameters):
        pass

    def get_model_params(self):
        if self.id == 0:
            return [p.data for p in self.model.parameters() if p.requires_grad]
        else:
            logging.error('Method get_model_params is supposed to be used by server.')
            sys.exit(1)

    def get_server_control_variates(self):
        if self.id == 0:
            return self.server_controls
        else:
            logging.error('Method get_server_control_variates is supposed to be used by server.')
            sys.exit(1)

    def set_server_model_params(self, server_model_parameters):
        if self.id != 0:
            for model_params, new_model_params in zip(self.server_model, server_model_parameters):
                model_params.data = new_model_params.data.to(self.device)
        else:
            logging.error('Method set_server_model_params is supposed to be used by clients.')
            sys.exit(1)

    def set_server_control_variates(self, server_controls):
        if self.id != 0:
            for control, new_control in zip(self.server_controls, server_controls):
                control.data = new_control.data.to(self.device)
        else:
            logging.error('Method set_server_control_variates is supposed to be used by clients.')
            sys.exit(1)

    def get_delta_model_params(self):
        if self.id != 0:
            return [data.to("cpu") for data in self.delta_model]
        else:
            logging.error('Method get_delta_model_params is supposed to be used by clients.')
            sys.exit(1)

    def get_delta_control_variates(self):
        if self.id != 0:
            return [data.to("cpu") for data in self.delta_controls]
        else:
            logging.error('Method get_delta_control_variates is supposed to be used by clients.')
            sys.exit(1)

    def train_epoch(self, criterion, opt, epoch, round_idx):
        train_loss = 0.0
        self.model.train()
        for i, batch in enumerate(self.train_data):
            src = batch["input"].float()
            src = src.to(self.device)
            src = self.encoder(src)
            trg = batch["target"].float()
            trg = trg.to(self.device)
            trg = self.encoder(trg)
            out = self.model(src, src_mask=self.mask)

            opt.zero_grad()
            assert out.size(1) == trg.size(1)
            loss = criterion(out, trg)
            loss.backward()
            opt.step(self.server_controls, self.controls)
            train_loss += loss.item()

        train_loss = train_loss / len(self.train_data)
        self.train_loss_list.append(train_loss)

        logging.info('Trainer_ID {}. Local Training Epoch: {} \tTrain Loss: {:.6f}'.format(self.id, epoch, train_loss))

        if self.val_data is None:
            if train_loss < self.min_loss:
                self.min_loss = train_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_optimizer = deepcopy(opt.state_dict())
                self.best_comm_round = round_idx + 1
                self.best_epoch_in_round = epoch
        else:
            self.validate_epoch(criterion, opt, epoch, round_idx)

    def validate_epoch(self, criterion, opt, epoch, round_idx):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_data):
                src = batch["input"].float()
                src = src.to(self.device)
                src = self.encoder(src)
                trg = batch["target"].float()
                trg = trg.to(self.device)
                trg = self.encoder(trg)
                out = self.model(src, src_mask=self.mask)

                loss = criterion(out, trg)
                val_loss += loss.item()

        val_loss = val_loss / len(self.val_data)
        self.val_loss_list.append(val_loss)
        logging.info('Trainer_ID {}. Local Training Epoch: {} \tValidation Loss: {:.6f}'.format(self.id, epoch, val_loss))

        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.best_model = deepcopy(self.model.state_dict())
            self.best_optimizer = deepcopy(opt.state_dict())
            self.best_comm_round = round_idx + 1
            self.best_epoch_in_round = epoch

    def train(self, round_idx):
        self.model.to(self.device)
        self.encoder.to(self.device)

        start = time.perf_counter()
        logging.info("-----START TRAINING THE TRANSFORMER-----")
        model_opt = SCAFFOLDOptimizer(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['L'])
        criterion = nn.MSELoss()
        for epoch in range(1, self.config["trans_num_epoch"] + 1):
            logging.info("Training local epoch {}...".format(epoch))
            self.train_epoch(criterion,
                             model_opt,
                             epoch,
                             round_idx)

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get client new controls
        new_controls = [torch.zeros_like(p.data, device=self.device) for p in self.model.parameters() if p.requires_grad]
        opt = 2
        if opt == 1:
            pass
        if opt == 2:
            for server_control, control, new_control, delta in zip(self.server_controls,
                                                                   self.controls,
                                                                   new_controls,
                                                                   self.delta_model):
                a = 1 / (math.ceil(self.num_train_samples / self.config['batch_size']) * self.config['lr'])
                new_control.data = control.data - server_control.data - delta.data * a

        # get controls differences
        for control, new_control, delta in zip(self.controls, new_controls, self.delta_controls):
            delta.data = new_control.data - control.data
            control.data = new_control.data

        logging.info("-----COMPLETED TRAINING THE TRANSFORMER-----")
        self.config["trans_train_time_round_" + str(round_idx)] = (time.perf_counter() - start) / 60

        torch.save(self.best_model, self.config["checkpoint_dir"] + "transformer_model.pt")
        torch.save(self.best_optimizer, self.config["checkpoint_dir"] + "transformer_opt.pt")
        self.config["best_trans_model"] = {"CommRound": self.best_comm_round, "Epoch": self.best_epoch_in_round}
        self.save_loss(round_idx)
        self.train_loss_list = list()
        self.val_loss_list = list()

    def test(self, test_data, device, args):
        pass

    def save_loss(self, round_idx):
        loss_file = self.config["result_dir"] + 'transformer_epoch_loss.csv'
        if self.val_data is not None:
            df_loss = pd.DataFrame([[round_idx+1, epoch+1, self.train_loss_list[epoch], self.val_loss_list[epoch]] for epoch in range(len(self.train_loss_list))])
            df_loss.to_csv(loss_file,
                           mode='a',
                           index=False,
                           header=False if os.path.exists(loss_file) else ['CommRound', 'LocalEpoch','TrainingLoss','ValidationLoss'])
        else:
            df_loss = pd.DataFrame([[round_idx+1, epoch+1, self.train_loss_list[epoch]] for epoch in range(len(self.train_loss_list))])
            df_loss.to_csv(loss_file,
                           mode='a',
                           index=False,
                           header=False if os.path.exists(loss_file) else ['CommRound', 'LocalEpoch','TrainingLoss'])

    def client_plot_loss(self):
        loss_file = self.config["result_dir"] + 'transformer_epoch_loss.csv'
        model_type = "TRANSFORMER" if self.config['model'] == 'transformer' else 'FNET_HYBRID'
        df_loss = pd.read_csv(loss_file)
        df_plot = df_loss[df_loss['LocalEpoch'] == self.config['trans_num_epoch']] # select last local epoch loss to plot
        rounds = df_plot.loc[:,'CommRound']
        train_loss = df_plot.loc[:, 'TrainingLoss']
        plt.plot(rounds, train_loss, 'g', label='Training loss')
        if self.val_data is not None:
            val_loss = df_plot.loc[:, 'ValidationLoss']
            plt.plot(rounds, val_loss, 'b', label='Validation loss')
            plt.title('{}. ID {}: Training and Validation Loss'.format(model_type, self.id))
        else:
            plt.title('{}. ID {}: Training Loss'.format(model_type, self.id))
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.config["result_dir"], self.config['model'] + "_loss.png"), dpi=300)
        plt.close()

    def get_total_training_time(self, num_comm_rounds):
        self.config["total_trans_train_time"] = 0.0
        for round in range(num_comm_rounds):
            self.config["total_trans_train_time"] += self.config[f"trans_train_time_round_{round}"]

    def get_updated_config(self):
        return self.config

    def save_aggregated_model(self, round_idx):
        if self.id == 0:
            directory = self.config["server_model_dir"]
            self.config["last_aggregated_server_model"] = f"aggregated_transformer_r{round_idx}.pt"
            torch.save(self.model.state_dict(), directory + self.config["last_aggregated_server_model"])
        else:
            logging.error('Method save_aggregated_model() is supposed to be used on server after SCAFFOLD aggregation.')
            sys.exit(1)

    def _create_mask(self):
        mask = torch.ones(1, self.config["l_win"], self.config["l_win"])
        mask[:, self.config["pre_mask"]:self.config["post_mask"], :] = 0
        mask[:, :, self.config["pre_mask"]:self.config["post_mask"]] = 0
        mask = mask.float().masked_fill(mask == 0, float(
            "-inf")).masked_fill(mask == 1, float(0))
        return mask
