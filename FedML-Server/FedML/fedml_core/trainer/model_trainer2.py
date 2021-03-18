from abc import ABC, abstractmethod

# This abstract class is for additional models (LCHA, VAE - LSTM)
class ModelTrainer2(ABC):
    def __init__(self, model):
        self.model = model
        self.id = 0

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_pca, y_train, args):
        pass

    @abstractmethod
    def test(self, test_pca, y_test):
        pass
