# Author: tangweize
# Date: 2025/3/10 17:52
# Description: 
# Data Studio Task:

import tensorflow as tf
from tensorflow.keras import layers, Model

import tensorflow as tf
from tensorflow.keras import layers, Model


def Custom_Model(Model):
    def __init__(self, preprocess_model, units):
        super().__init__()
        self.preprocess_model = preprocess_model
        self.dnn = keras.Sequential()
        for unint in units:
            self.dnn.add(
                layers.Dense(unint, activation='relu'
                             )
            self.dnn.add(layers.Dense(1))

    def call(self, inputs):
        x = self.preprocess_model(inputs)
        x = self.dnn(x)
        return x

    def tain_step(self, inputs, label):
        with tf.GradientTape() as tape:
            predict = self(inputs)
            self.loss




