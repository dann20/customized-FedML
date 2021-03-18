import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense

class LCHA:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(20, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

