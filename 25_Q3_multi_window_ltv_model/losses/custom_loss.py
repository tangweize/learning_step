# Author: tangweize
# Date: 2025/6/18 20:29
# Description: 
# Data Studio Task:


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

class MSELTVLoss(tf.keras.losses.Loss):
    def __init__(self, mode='mse', normalize=False, name=None):
        self.mode = mode.lower()
        self.normalize = normalize
        if name is None:
            name = f"{self.mode}_ltv_loss"
        super().__init__(name=name)


    def call(self, y_true_packed, y_pred):
        mode = self.mode
        y_true = y_true_packed[:, 0]
        loss = tf.reduce_mean(tf.square(y_true - y_pred))

        return loss


class UnifiedLTVLoss(tf.keras.losses.Loss):
    def __init__(self, mode='delta', normalize=False, name=None, regular = 0.01):
        """
        支持模式：
        - 'delta':         预测增量 LTV（7d - 1h）
        - 'log':           预测 log(1 + 7d LTV)
        - 'log_delta':     预测 log(7d - 1h)
        - 'mse':           普通均方误差
        - 'mae':           平均绝对误差
        - 'mape':          平均绝对百分比误差
        - 'binary':        二分类交叉熵（sigmoid + binary_crossentropy）

        :param normalize: 是否对 loss 进行归一化（loss / stop_gradient(loss)）
        """
        self.mode = mode.lower()
        self.normalize = normalize
        self.regular = regular
        self.p = 1.5
        if name is None:
            name = f"{self.mode}_ltv_loss"
        super().__init__(name=name)

    def normalize_loss(self, loss):
        if self.normalize:
            norm_factor = tf.stop_gradient(loss) + 1e-8  # 防止除零
            return loss / norm_factor
        return loss

    def call(self, y_true_packed, y_pred):
        mode = self.mode

        # 将所有 y_true_packed[:, i] 转换为 (-1, 1) 的形状
        y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))
        if mode == 'delta':
            ltv_1h = tf.reshape(y_true_packed[:, 1], (-1, 1))  # Reshape to (-1, 1)
            delta_true = tf.maximum(y_true - ltv_1h, 0.0)
            loss = tf.reduce_mean(tf.square(delta_true - y_pred))

        elif mode == 'log':
            # 小于0的y_true先置为0，然后再 log1p
            y_true_clipped = tf.maximum(y_true, 0.0)
            y_true_log = tf.math.log1p(y_true_clipped)
            loss = tf.reduce_mean(tf.square(y_true_log - y_pred))

        elif mode == 'log_delta':
            ltv_1h = tf.reshape(y_true_packed[:, 1], (-1, 1))  # Reshape to (-1, 1)
            delta_true = tf.maximum(y_true - ltv_1h, 1e-5)
            delta_pred = tf.maximum(y_pred, 1e-5)
            loss = tf.reduce_mean(tf.square(tf.math.log(delta_true) - tf.math.log(delta_pred)))

        elif mode == 'mse':
            loss = tf.reduce_mean(tf.square(y_true - y_pred))

        elif mode == 'delta_regular':
            ltv_1h = tf.reshape(y_true_packed[:, 1], (-1, 1))  # Reshape to (-1, 1)
            delta_true = tf.maximum(y_true - ltv_1h, 0.0)
            loss = tf.reduce_mean(tf.square(delta_true - y_pred)) + self.regular * tf.abs(tf.reduce_mean(y_pred) - tf.reduce_mean(delta_true))

        elif mode == 'mae':
            loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        elif mode == 'mape':
            loss = tf.reduce_mean(tf.abs((y_true - y_pred) / tf.maximum(y_true, 1e-5)))

        elif mode == 'delta_mape':
            ltv_1h = tf.reshape(y_true_packed[:, 1], (-1, 1))
            delta_true = tf.maximum(y_true - ltv_1h, 1e-5)
            loss = tf.reduce_mean(tf.abs((delta_true - y_pred) / delta_true))



        elif mode == 'tweedie':

            y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1e6)
            term2 = tf.math.pow(y_pred, 2 - self.p) / (2 - self.p)
            # tf.print(y_pred)
            # tf.print(y_true)
            term1 = y_true * tf.math.pow(y_pred, 1 - self.p) / (1 - self.p)
            temp_loss = (term2 - term1)
            # debug
            # tf.print(temp_loss)
            loss = tf.reduce_mean(temp_loss)

        elif mode == 'delta_tweedie':
            y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=1e6)
            ltv_1h = tf.reshape(y_true_packed[:, 1], (-1, 1))  # Reshape to (-1, 1)
            y_true = tf.maximum(y_true - ltv_1h, 0)
            term2 = tf.math.pow(y_pred, 2 - self.p) / (2 - self.p)
            term1 = y_true * tf.math.pow(y_pred, 1 - self.p) / (1 - self.p)
            temp_loss = (term2 - term1)
            loss = tf.reduce_mean(temp_loss)


        elif mode == 'binary':
            y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))  # Reshape to (-1, 1)
            y_pred = tf.squeeze(y_pred, axis=-1)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True))

        else:
            raise ValueError(f"Unsupported loss mode: {mode}")

        return self.normalize_loss(loss)


class UnifiedLTVsimple(tf.keras.losses.Loss):
    def __init__(self, mode='delta', normalize=False, name=None):
        """
        支持模式：
        - 'delta':         预测增量 LTV（7d - 1h）
        - 'log':           预测 log(1 + 7d LTV)
        - 'log_delta':     预测 log(7d - 1h)
        - 'mse':           普通均方误差
        - 'mae':           平均绝对误差
        - 'mape':          平均绝对百分比误差
        - 'binary':        二分类交叉熵（sigmoid + binary_crossentropy）

        :param normalize: 是否对 loss 进行归一化（loss / stop_gradient(loss)）
        """
        self.mode = mode.lower()
        self.normalize = normalize
        if name is None:
            name = f"{self.mode}_ltv_loss"
        super().__init__(name=name)

    def normalize_loss(self, loss):
        if self.normalize:
            norm_factor = tf.stop_gradient(loss) + 1e-8  # 防止除零
            return loss / norm_factor
        return loss

    def call(self, y_true_packed, y_pred):
        mode = self.mode

        if mode == 'mse':
            y_true = tf.reshape(y_true_packed, (-1, 1))  # Reshape to (-1, 1)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        elif mode == 'log':
            y_true_clipped = tf.maximum(y_true_packed, 0.0)
            y_true_log = tf.math.log1p(y_true_clipped)
            loss = tf.reduce_mean(tf.square(y_true_log - y_pred))
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")

        return self.normalize_loss(loss)