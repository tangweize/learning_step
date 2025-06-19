# Author: tangweize
# Date: 2025/6/18 20:29
# Description: 
# Data Studio Task:


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras



class UnifiedLTVLoss(tf.keras.losses.Loss):
    def __init__(self, mode='delta', name=None):
        """
        支持模式：
        - 'delta':    预测增量 LTV（7d - 1h）
        - 'log':      预测 log(1 + 7d LTV)
        - 'log_delta':预测 log(7d - 1h)
        - 'mse':      普通均方误差
        - 'mae':      平均绝对误差
        - 'mape':     平均绝对百分比误差
        """
        self.mode = mode.lower()
        if name is None:
            name = f"{self.mode}_ltv_loss"
        super().__init__(name=name)

    def call(self, y_true_packed, y_pred):
        mode = self.mode

        if mode == 'delta':
            y_true = y_true_packed[:, 0]
            ltv_1h = y_true_packed[:, 1]
            delta_true = y_true - ltv_1h
            return tf.reduce_mean(tf.square(delta_true - y_pred))

        elif mode == 'log':
            y_true_log = tf.math.log1p(y_true_packed[:, 0])
            y_pred_log = tf.math.log1p(y_pred)
            return tf.reduce_mean(tf.square(y_true_log - y_pred_log))

        elif mode == 'log_delta':
            y_true = y_true_packed[:, 0]
            ltv_1h = y_true_packed[:, 1]
            delta_true = tf.maximum(y_true - ltv_1h, 1e-5)
            delta_pred = tf.maximum(y_pred, 1e-5)
            return tf.reduce_mean(tf.square(tf.math.log(delta_true) - tf.math.log(delta_pred)))

        elif mode == 'mse':
            y_true = y_true_packed[:, 0]
            return tf.reduce_mean(tf.square(y_true - y_pred))

        elif mode == 'mae':
            y_true = y_true_packed[:, 0]
            return tf.reduce_mean(tf.abs(y_true - y_pred))

        elif mode == 'mape':
            y_true = y_true_packed[:, 0]
            return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.maximum(y_true, 1e-5)))

        else:
            raise ValueError(f"Unsupported loss mode: {mode}")