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
                input_tensor = tf.cast(input_tensor, tf.float32)  # 显式转 float
                input_cast = tf.maximum(input_tensor, 0.0)
                normalized_feature = tf.math.log1p(input_cast + 1) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                # temp_feature = tf.expand_dims(normalized_feature)
                processed_dense_features.append(normalized_feature)
            elif field in self.dense_price_features or field in self.dense_duration_features:
                input_tensor = tf.cast(input_tensor, tf.float32)  # 显式转 float
                input_cast = tf.maximum(input_tensor, 0.0)
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
                self.field2embedding[feature] = layers.Embedding(input_dim=500, output_dim=4)
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

    # def build(self, input_shape):
    #     dummy_inputs = {
    #         k: tf.keras.Input(shape=v, dtype=tf.float32)  # 注意这里直接用 v
    #         for k, v in input_shape.items()
    #     }
    #     # self(dummy_inputs)  # 调用 call 构建所有子层
    #     super().build(input_shape)



    def call(self, inputs):
        dense_tensor = self.process_dense_layer(inputs)
        emb_tesor = self.process_emb_layer(inputs)

        input_tensor = layers.Concatenate()([dense_tensor, emb_tesor])

        sharebottom_out = self.sharebottom(input_tensor)

        outputs = [head(sharebottom_out) for head in self.hour2headnn]

        hour_idx = -1
        for i, v in enumerate(self.sparse_features):
            if v == self.hour_flag:
                hour_idx = tf.cast(tf.gather(inputs[self.sparse_group_name], indices=i, axis = 1) - 1, tf.int64)
                hour_idx = tf.clip_by_value(hour_idx, 0, self.num_heads - 1)

        outputs = self.concat_layer(outputs)
        selected = tf.gather(outputs, hour_idx, axis=1, batch_dims=1)
        selected = tf.expand_dims(selected, axis=1)  # 模拟 keep_dim=True 的效果
        return selected

    def train_step(self, train_data):
        inputs, label = train_data
        with tf.GradientTape() as tape:
            predict = self(inputs)

            losses = self.loss(label, predict)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(label, predict)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def predict(self, inputs, batch_size=None):
        return super().predict(inputs, batch_size=batch_size)

    def evaluate(self, test_dataset):

        hour_model_pred = {k: 0 for k in range(self.num_heads)}
        hour_model_true = {k: 0 for k in range(self.num_heads)}
        mode = self.loss.mode

        for batch in test_dataset:
            inputs, y_true_packed = batch
            pred = self(inputs)

            hour_idx = None

            if mode in ('delta', 'log_delta') :
                y_pred = pred + y_true_packed[:, 1:2]  # shape (B, 1)
                y_true = y_true_packed[:, 0]

            elif mode in ['mse', 'mae', 'mape', 'log']:
                y_pred = pred
                y_true = y_true_packed[:, 0]
            else:
                raise ValueError(f"Unsupported mode in evaluation: {mode}")




            for i, v in enumerate(self.sparse_features):
                if v == self.hour_flag:
                    hour_idx = tf.cast(tf.gather(inputs[self.sparse_group_name], indices=i, axis = 1) - 1, tf.int64)
                    hour_idx = tf.clip_by_value(hour_idx, 0, self.num_heads - 1)

            for head in range(self.num_heads):
                idxs = tf.where(tf.equal(hour_idx, head))[:, 0]
                if tf.size(idxs) == 0:
                    continue  # 该 head 在当前 batch 没有数据，跳过

                # 选出对应 head 的 pred 和 label
                head_pred = tf.gather(pred, idxs)
                head_true = tf.gather(y_true, idxs)  # 假设label[:, 0]是你想评估的目标

                hour_model_pred[head] += tf.reduce_sum(head_pred)
                hour_model_true[head] += tf.reduce_sum(head_true)


        return hour_model_pred, hour_model_true