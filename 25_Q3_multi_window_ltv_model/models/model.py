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
        return self.concat_layer(processed_dense_features)

import tensorflow as tf
from tensorflow.keras import layers


class Sparse_Process_Layer(layers.Layer):
    def __init__(self, sparse_group_name, user_sparse_features, emb_features):
        super().__init__()
        self.sparse_group_name = sparse_group_name
        self.user_sparse_features = user_sparse_features

        self.field2embedding = {}

        for feature in user_sparse_features:
            if feature in emb_features:
                self.field2embedding[feature] = layers.Embedding(input_dim=50, output_dim=4)
        self.concat_layer = layers.Concatenate()

    def call(self, inputs):
        processed_dense_features = []
        for field, input_tensor in inputs.items():
            if field == self.sparse_group_name:
                for idx, v in enumerate(self.user_sparse_features):
                    # 分头flag
                    if v == 'request_hour_diff':
                        continue
                        # input_tensor 是个二维数据 batch x n_features，其中 idx 对应特征列
                    target_feature = tf.gather(input_tensor, indices=idx, axis=1)
                    if v in self.field2embedding:
                        embed = self.field2embedding[v](target_feature)
                        processed = tf.reshape(embed, [-1, embed.shape[-1]])
                    else:
                        processed = tf.cast(target_feature, tf.float32)
                        processed = tf.expand_dims(processed, axis=1)
                    processed_dense_features.append(processed)

        return self.concat_layer(processed_dense_features)

class HEAD_DNN(layers.Layer):
    def __init__(self, units, activation = 'relu'):
        super().__init__()
        self.dnn = keras.Sequential([
            layers.Dense(unit, activation= activation) for unit in units
        ])

        # output
        self.dnn = layers.Dense(1, activation= activation)
    def call(self, x):
        return self.dnn(x)

class DNN(layers.Layer):
    def __init__(self, units, activation = 'relu'):
        super().__init__()
        self.dnn = keras.Sequential([
            layers.Dense(unit, activation= activation) for unit in units
        ])

    def call(self, x):
        return self.dnn(x)


class MULTI_HEAD_LTV_MODEL(keras.Model):
    def __init__(self,
                 num_heads,
                 units,
                 head_units,
                 dense_cnt_feature_name,
                 dense_price_feature_name,
                 dense_duration_feature_name,
                 sparse_group_name,
                 user_sparse_features,
                 emb_features,
                 hour_flag = 'request_hour_diff'):
        super().__init__()
        self.num_heads = num_heads
        self.process_dense_layer = Dense_Process_LOG_Layer(dense_cnt_feature_name, dense_price_feature_name, dense_duration_feature_name)
        self.process_emb_layer = Sparse_Process_Layer(sparse_group_name, user_sparse_features, emb_features)
        self.sparse_features = user_sparse_features
        self.sparse_group_name = sparse_group_name
        self.hour_flag = hour_flag

        self.sharebottom = DNN(units)
        self.hour2headnn = [ HEAD_DNN(head_units) for i in range(num_heads)]
        self.concat_layer = layers.Concatenate()

    def call(self, inputs):
        dense_tensor = self.process_dense_layer(inputs)
        emb_tesor = self.process_emb_layer(inputs)

        sharebottom = layers.Concatenate()([dense_tensor, emb_tesor])

        outputs = [head(sharebottom) for head in self.hour2headnn]

        hour_idx = -1
        for i, v in enumerate(self.sparse_features):
            if v == self.hour_flag:
                hour_idx = tf.cast(tf.gather(inputs[self.sparse_group_name], indices=i, axis = 1) - 1, tf.int64)
                hour_idx = tf.minimum(hour_idx, self.num_heads - 1)

        outputs = self.concat_layer(outputs)
        return tf.gather(outputs, hour_idx, axis = 1, batch_dims=1)