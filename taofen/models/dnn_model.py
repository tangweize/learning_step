import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

# 正则项 loss之和：  self.losses
# model 类的 loss 损失函数 
class Custom_Model(Model):
    def __init__(self, preprocess_model, units):
        super().__init__()
        self.preprocess_model = preprocess_model
        self.dnn = keras.Sequential()
        for unint in units:
            self.dnn.add(
                layers.Dense(unint, activation='relu')
            )
        self.dnn.add(layers.Dense(1))

    def call(self, inputs):
        x = self.preprocess_model(inputs)
        x = self.dnn(x)
        return tf.sigmoid(x)

    def tain_step(self, inputs, label):


        with tf.GradientTape() as tape:
            predict = self(inputs)
            losses = self.loss(label,predict)

        trainable_vars = self.trainable_vars
        gradients = tape.GradientTape(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        results = {m.name: m.result() for m in self.metrics}
        return results