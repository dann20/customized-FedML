import time
import logging
import os

import pandas as pd

import torch
from torch import nn
from torch import optim

from FedML.fedml_core.trainer.model_trainer import ModelTrainer

class FedAVGTransformerTrainer(ModelTrainer):
    def __init__(self, id, autoencoder_model, transformer_model, train_data, device, config):
        self.id = id    # id = 0 denotes server, denotes clients otherwise
        self.model = transformer_model
        if autoencoder_model != None:
            self.encoder = autoencoder_model.encoder
            self.encoder.eval()
        self.train_data = train_data
        self.device = device
        self.config = config
        self.min_loss = float('inf')
        self.epoch_loss = list()
        self.best_model = None
        self.mask = self._create_mask()

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

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
            opt.step()
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
        criterion = nn.MSELoss()
        for epoch in range(self.config["trans_num_epoch"]):
            self.train_epoch(criterion,
                             model_opt,
                             epoch,
                             round_idx)
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

    def get_len_data(self):
        return len(self.train_data.dataset)

    def save_aggregated_model(self, round_idx):
        if self.id == 0:
            directory = self.config["server_model_dir"]
            self.config["last_aggregated_server_model"] = f"aggregated_transformer_r{round_idx}.pt"
            torch.save(self.model.state_dict(), directory + self.config["last_aggregated_server_model"])
        else:
            raise Exception('Method save_aggregated_model() is supposed to be used on server after FedAvg aggregation')

    def _create_mask(self):
        mask = torch.ones(1, self.config["l_win"], self.config["l_win"])
        mask[:, self.config["pre_mask"]:self.config["post_mask"], :] = 0
        mask[:, :, self.config["pre_mask"]:self.config["post_mask"]] = 0
        mask = mask.float().masked_fill(mask == 0, float(
            "-inf")).masked_fill(mask == 1, float(0))
        return mask
