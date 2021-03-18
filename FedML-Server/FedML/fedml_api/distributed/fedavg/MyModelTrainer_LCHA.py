import logging
import tensorflow as tf

try:
    from fedml_core.trainer.model_trainer2 import ModelTrainer2
except ImportError:
    from FedML.fedml_core.trainer.model_trainer2 import ModelTrainer2

class MyModelTrainerLCHA(ModelTrainer2):
    def __init__(self, model, args):
        super().__init__(model)
        loss='sparse_categorical_crossentropy'
        metrics = ['accuracy']
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
        self.model.compile(loss=loss, optimizer=optimizer, metrics = metrics)

    def get_model_params(self):
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        self.model.set_weights(model_parameters)

    def train(self, train_pca, y_train, args):
        # model = self.model
        self.model.fit(train_pca, y_train, epochs = args.epochs, verbose = 0, batch_size = args.batch_size)

    def test(self, test_pca, y_test):
        [loss, acc] = self.model.evaluate(test_pca, y_test)
        return loss,acc


