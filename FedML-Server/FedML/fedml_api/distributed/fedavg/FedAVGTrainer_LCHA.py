import numpy as np

class FedAVGTrainer(object):
    def __init__(self, client_index, train_pca, train_label, test_pca, test_label, args, model_trainer):
        self.trainer = model_trainer
        self.train_pca = train_pca
        self.train_label = train_label
        self.test_pca = test_pca
        self.test_label = test_label
        self.local_sample_number = self.train_pca.shape[0]
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def train(self):
        self.trainer.train(self.train_pca, self.train_label, self.args)
        weights = self.trainer.get_model_params()
        return weights, self.local_sample_number

    def test(self):
        # train data
        train_loss,train_acc = self.trainer.test(self.train_pca, self.train_label)

        # test data
        test_loss, test_acc = self.trainer.test(self.test_pca, self.test_label)
        return train_acc, train_loss, test_acc, test_loss
