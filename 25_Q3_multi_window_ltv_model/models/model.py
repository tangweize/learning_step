# Author: tangweize
# Date: 2025/6/17 20:03
# Description: 
# Data Studio Task:

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf



class Dense_Process_LOG_Layer(layers.Layer):
    def __init__(self, dense_cnt_features, dense_price_features, dense_duration_features):
        super().__init__()
        self.dense_cnt_features = dense_cnt_features
        self.dense_price_features = dense_price_features
        self.dense_duration_features = dense_duration_features
        self.concat_layer = layers.Concatenate()
    def call(self, inputs):
        processed_dense_features = []
        for field, input_tensor in inputs.items():
            if field in self.dense_cnt_features:
                input_cast = tf.maximum(input_tensor, 0)
                normalized_feature = tf.math.log1p(input_cast + 1) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                # temp_feature = tf.expand_dims(normalized_feature)
                processed_dense_features.append(normalized_feature)
            elif field in self.dense_price_features or field in self.dense_duration_features:
                input_cast = tf.maximum(input_tensor, 0)
                normalized_feature = tf.math.log1p(input_cast + 1) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
                # temp_feature = tf.expand_dims(normalized_feature)
                processed_dense_features.append(normalized_feature)
        print(processed_dense_features[0].shape)
        return self.concat_layer(processed_dense_features)