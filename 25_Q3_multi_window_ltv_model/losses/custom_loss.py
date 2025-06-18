# Author: tangweize
# Date: 2025/6/18 20:29
# Description: 
# Data Studio Task:


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras



def residual_ltv_loss(y_true, y_pred, hour1_ltv):
    # 真实的增量
    residual_true = y_true - hour1_ltv
    # 回归残差
    return tf.reduce_mean(tf.square(residual_true - y_pred))



def log_residual_ltv_loss(y_true, y_pred, hour1_ltv, c=1.0):
    residual_true = y_true - hour1_ltv
    log_true = tf.math.log(tf.maximum(residual_true + c, 1e-6))
    return tf.reduce_mean(tf.square(log_true - y_pred))



def full_ltv_loss(y_true, y_pred, hour1_ltv):
    return tf.reduce_mean(tf.square(y_true - y_pred))




#  sum(a) - sum(b)  < 0