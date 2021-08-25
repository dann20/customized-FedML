import time
import logging

import torch
from torch import nn
from torch import optim

from FedML.fedml_core.trainer.model_trainer import ModelTrainer

class TransformerTrainer(ModelTrainer):
    def __init__(self, autoencoder_model, transformer_model, train_data, device, config):
        self.id = 0
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
        self.round_idx = 0

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train_epoch(self, criterion, opt, epoch):
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
                                                                                         sum(self.epoch_loss) / len(self.epoch_loss)))
        if self.epoch_loss[-1] < self.min_loss:
            torch.save(self.model.state_dict(),
                       self.config["checkpoint_dir"] + f"best_trans_r{self.round_idx}_e{epoch}.pth")
            torch.save(opt.state_dict(),
                       self.config["checkpoint_dir"] + f"optimizer_trans_r{self.round_idx}_e{epoch}.pth")
            self.min_loss = self.epoch_loss[-1]
            self.best_model = f"best_trans_r{self.round_idx}_e{epoch}.pth"

    def train(self):
        start = time.time()
        logging.info("-----START TRAINING THE TRANSFORMER-----")
        self.model.float()
        model_opt = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(self.config["trans_num_epoch"]):
            self.train_epoch(criterion,
                             model_opt,
                             epoch)
        logging.info("-----COMPLETED TRAINING THE TRANSFORMER-----")
        self.config["best_trans_model_round_" + str(self.round_idx)] = self.best_model
        self.config["trans_train_time_round_" + str(self.round_idx)] = (time.time() - start) / 60

    def test(self, test_data, device, args):
        pass

    def update_comm_round(self):
        self.round_idx += 1

    def get_updated_config(self):
        return self.config

    def get_len_data(self):
        return len(self.train_data.dataset)

    def _create_mask(self):
        mask = torch.ones(1, self.config["l_win"], self.config["l_win"])
        mask[:, self.config["pre_mask"]:self.config["post_mask"], :] = 0
        mask[:, :, self.config["pre_mask"]:self.config["post_mask"]] = 0
        mask = mask.float().masked_fill(mask == 0, float(
            "-inf")).masked_fill(mask == 1, float(0))
        return mask
