import time
import logging
import os
import math

import pandas as pd

import torch
from torch import nn
from torch import optim

from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_api.distributed.scaffold.SCAFFOLDOptimizer import SCAFFOLDOptimizer

class SCAFFOLDTransformerTrainer(ModelTrainer):
    def __init__(self, id, autoencoder_model, transformer_model, train_data, device, config):
        self.id = id    # id = 0 denotes server, denotes clients otherwise
        self.model = transformer_model
        if autoencoder_model != None:
            self.encoder = autoencoder_model.encoder
            self.encoder.eval()
        self.train_data = train_data
        self.num_train_samples = len(self.train_data.dataset) if self.id != 0 else None
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.epoch_loss = list()
        self.best_model = None
        self.mask = self._create_mask()

        if self.id == 0:
            self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        else:
            self.server_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.delta_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
            self.delta_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

    def set_model_params(self, model_parameters):
        pass

    def get_model_params(self):
        if self.id == 0:
            return [p.data for p in self.model.parameters() if p.requires_grad]
        else:
            raise Exception('Method get_model_params is supposed to be used by server.')

    def get_server_control_variates(self):
        if self.id == 0:
            return self.server_controls
        else:
            raise Exception('Method get_server_control_variates is supposed to be used by server.')

    def set_server_model_params(self, server_model_parameters):
        if self.id != 0:
            for model_params, new_model_params in zip(self.server_model, server_model_parameters):
                model_params.data = new_model_params.data
        else:
            raise Exception('Method set_server_model_params is supposed to be used by clients.')

    def set_server_control_variates(self, server_controls):
        if self.id != 0:
            for control, new_control in zip(self.server_controls, server_controls):
                control.data = new_control.data
        else:
            raise Exception('Method set_server_control_variates is supposed to be used by clients.')

    def get_delta_model_params(self):
        if self.id != 0:
            return self.delta_model
        else:
            raise Exception('Method get_delta_model_params is supposed to be used by clients.')

    def get_delta_control_variates(self):
        if self.id != 0:
            return self.delta_controls
        else:
            raise Exception('Method get_delta_control_variates is supposed to be used by clients.')

    def train_epoch(self, criterion, opt, epoch, round_idx):
        self.model.train()
        self.model.to(self.device)
        encoder = self.encoder
        encoder.to(self.device)
        batch_loss = list()
        for i, batch in enumerate(self.train_data):
            src = batch["input"].float()
            src.to(self.device)
            src = encoder(src)
            trg = batch["target"].float()
            trg.to(self.device)
            trg = encoder(trg)
            out = self.model(src, src_mask=self.mask)

            opt.zero_grad()
            assert out.size(1) == trg.size(1)
            loss = criterion(out, trg)
            loss.backward()
            opt.step(self.server_controls, self.controls)
            batch_loss.append(loss.item())

        if len(batch_loss) > 0:
            self.epoch_loss.append(sum(batch_loss)/len(batch_loss))
            logging.info('Trainer_ID {}. Local Training Epoch: {} \tTotal Loss: {:.6f}'.format(self.id,
                                                                                               epoch,
                                                                                               self.epoch_loss[-1]))

        if self.epoch_loss[-1] < self.min_loss:
            torch.save(self.model.state_dict(),
                       self.config["checkpoint_dir"] + f"best_trans_r{round_idx}_e{epoch}.pt")
            torch.save(opt.state_dict(),
                       self.config["checkpoint_dir"] + f"optimizer_trans_r{round_idx}_e{epoch}.pt")
            self.min_loss = self.epoch_loss[-1]
            self.best_model = f"best_trans_r{round_idx}_e{epoch}.pt"

    def train(self, round_idx):
        start = time.time()
        logging.info("-----START TRAINING THE TRANSFORMER-----")
        self.model.float()
        model_opt = optim.Adam(self.model.parameters())
        model_opt = SCAFFOLDOptimizer(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['L'])
        criterion = nn.MSELoss()
        for epoch in range(self.config["trans_num_epoch"]):
            self.train_epoch(criterion,
                             model_opt,
                             epoch,
                             round_idx)

        # get model difference
        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get client new controls
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
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
        self.config["best_trans_model_round_" + str(round_idx)] = self.best_model
        self.config["trans_train_time_round_" + str(round_idx)] = (time.time() - start) / 60
        self.save_loss(round_idx)
        self.epoch_loss = list()

    def test(self, test_data, device, args):
        pass

    def save_loss(self, round_idx):
        df_loss = pd.DataFrame([[round_idx+1, i+1, self.epoch_loss[i]] for i in range(len(self.epoch_loss))])
        loss_file = self.config["result_dir"] + 'transformer_epoch_loss.csv'
        df_loss.to_csv(loss_file,
                       mode='a',
                       index=False,
                       header=False if os.path.exists(loss_file) else ['CommRound', 'LocalEpoch','Loss'])

    def get_updated_config(self):
        return self.config

    def save_aggregated_model(self, round_idx):
        if self.id == 0:
            directory = self.config["server_model_dir"]
            self.config["last_aggregated_server_model"] = f"aggregated_transformer_r{round_idx}.pth"
            torch.save(self.model.state_dict(), directory + self.config["last_aggregated_server_model"])
        else:
            raise Exception('Method save_aggregated_model() is supposed to be used on server after SCAFFOLD aggregation.')

    def _create_mask(self):
        mask = torch.ones(1, self.config["l_win"], self.config["l_win"])
        mask[:, self.config["pre_mask"]:self.config["post_mask"], :] = 0
        mask[:, :, self.config["pre_mask"]:self.config["post_mask"]] = 0
        mask = mask.float().masked_fill(mask == 0, float(
            "-inf")).masked_fill(mask == 1, float(0))
        return mask
