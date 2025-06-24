# Author: tangweize
# Date: 2025/6/17 20:03
# Description: 
# Data Studio Task:

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
                    if v != 'request_hour_diff':
                        # continue
                        # # input_tensor 是个二维数据 batch x n_features，其中 idx 对应特征列
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
        self.dnn.add(
            layers.Dense(1)
        )
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
        self.hour2headnn = [ HEAD_DNN(head_units) for i in range(self.num_heads)]
        self.concat_layer = layers.Concatenate()


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
        print(label)
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


    # 存储 每条样本的 pred 和 true，并分别保存为各个head；
    # 返回： pred:[],  true:[]
    def predict_head_score(self, test_dataset):
        hour_model_pred = {k: [] for k in range(self.num_heads)}
        hour_model_true = {k: [] for k in range(self.num_heads)}
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
            elif mode in ['binary']:
                y_pred = tf.sigmoid(pred)
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
                # print(y_pred.shape, y_true.shape) # (2048, 1) (432,)
                head_pred = tf.gather(y_pred, idxs)
                head_true = tf.gather(y_true, idxs)

                hour_model_pred[head].append(head_pred)
                hour_model_true[head].append(head_true)

        # 处理成 一维tensor

        for head in range(self.num_heads):
            hour_model_pred[head] = tf.concat(hour_model_pred[head], axis =  -1)
            hour_model_true[head] = tf.concat(hour_model_true[head], axis =  -1)


        return hour_model_pred, hour_model_true
    def evaluate_exp(self, test_dataset):
        # 计算各个头的期望的 bias

        head_pred, head_true = self.predict_head_score(test_dataset)

        mape = {}
        for head in head_pred.keys():
            # 对每个head的所有结果 求和
            head_pred[head] = tf.reduce_sum(head_pred[head]).numpy()
            head_true[head] = tf.reduce_sum(head_true[head]).numpy()


            pred = head_pred[head]
            true = head_true[head]
            mape[head] = (pred - true) / (true + 1.0)
            mape[head] = round(mape[head], 2)


        return {
            'pred_sum': head_pred,
            'true_sum': head_true,
            'bias' : mape
        }




    def evaluate_rank(self, test_dataset):

        head_pred, head_true = self.predict_head_score(test_dataset)

        rank_res = {}
        for head in head_pred.keys():
            head_name = f"{head + 1}_h rank_score"
            pred = head_pred[head]
            true = head_true[head]
            rank_res[head_name] = self.calculate_area_under_gain_curve(pred, true, head_name)


        return rank_res


    def calculate_area_under_gain_curve(self, pred_list, true_list, head_name=""):
        # 将零维张量列表转换为一维 NumPy 数组
        pred = np.array([p.numpy() for p in pred_list])
        true = np.array([t.numpy() for t in true_list])

        # 创建 DataFrame
        df = pd.DataFrame({'pred': pred, 'true': true})

        # 根据预测值进行排序
        df = df.sort_values(by='pred', ascending=False)

        # 计算累积百分比
        df['cumulative_percentage_customers'] = np.arange(1, len(df) + 1) / len(df)
        df['cumulative_percentage_ltv'] = df['true'].cumsum() / df['true'].sum()

        # 计算增益曲线下面积
        area = np.trapz(df['cumulative_percentage_ltv'], df['cumulative_percentage_customers'])

        # 绘制增益图
        plt.figure(figsize=(10, 6))
        plt.plot(df['cumulative_percentage_customers'], df['cumulative_percentage_ltv'], label="Gain Curve")
        plt.xlabel('Cumulative Percentage of Customers')
        plt.ylabel('Cumulative Percentage of Total LTV')
        plt.title(f'{head_name} Gain Chart')
        plt.legend()
        plt.grid(True)
        plt.show()

        return area



class MMOE(keras.Model):
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
                 hour_flag='request_hour_diff',
                 num_experts=4):  # 🔄 新增参数：expert 数量
        super().__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.hour_flag = hour_flag
        self.sparse_features = user_sparse_features
        self.sparse_group_name = sparse_group_name

        # 特征处理层
        self.process_dense_layer = Dense_Process_LOG_Layer(dense_cnt_feature_name, dense_price_feature_name, dense_duration_feature_name)
        self.process_emb_layer = Sparse_Process_Layer(sparse_group_name, user_sparse_features, emb_features)

        # MMoE 结构
        self.experts = [DNN(units) for _ in range(num_experts)]  # 多个专家网络
        self.gates = [layers.Dense(num_experts, activation='softmax') for _ in range(num_heads)]  # 每个任务一个门控
        self.towers = [HEAD_DNN(head_units) for _ in range(num_heads)]  # 每个任务一个输出层
        self.concat_layer = layers.Concatenate()

    def call(self, inputs):
        dense_tensor = self.process_dense_layer(inputs)
        emb_tensor = self.process_emb_layer(inputs)
        input_tensor = tf.concat([dense_tensor, emb_tensor], axis=1)

        # 每个 expert 输出 shape: (batch_size, expert_dim)
        expert_outputs = [expert(input_tensor) for expert in self.experts]  # list of (B, dim)
        expert_stack = tf.stack(expert_outputs, axis=1)  # shape (B, num_experts, dim)

        # 每个任务的 gate 输出权重 shape: (B, num_experts)，然后与 expert_stack 做加权
        task_outputs = []
        for i in range(self.num_heads):
            gate_weights = self.gates[i](input_tensor)  # shape (B, num_experts)
            gate_weights = tf.expand_dims(gate_weights, axis=-1)  # shape (B, num_experts, 1)
            weighted_expert_output = tf.reduce_sum(expert_stack * gate_weights, axis=1)  # shape (B, dim)
            task_output = self.towers[i](weighted_expert_output)  # shape (B, 1)
            task_outputs.append(task_output)

        outputs = self.concat_layer(task_outputs)  # shape (B, num_heads)

        # 动态选择当前样本属于哪一 hour_head
        hour_idx = -1
        for i, v in enumerate(self.sparse_features):
            if v == self.hour_flag:
                hour_idx = tf.cast(tf.gather(inputs[self.sparse_group_name], indices=i, axis=1) - 1, tf.int64)
                hour_idx = tf.clip_by_value(hour_idx, 0, self.num_heads - 1)

        selected = tf.gather(outputs, hour_idx, axis=1, batch_dims=1)
        selected = tf.expand_dims(selected, axis=1)
        return selected

    def train_step(self, train_data):
        inputs, label = train_data
        print(label)
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

    def evaluate_exp(self, test_dataset):

        hour_model_pred = {k: 0 for k in range(self.num_heads)}
        hour_model_true = {k: 0 for k in range(self.num_heads)}
        mode = self.loss.mode

        for batch in test_dataset:
            inputs, y_true_packed = batch
            pred = self(inputs)

            hour_idx = None

            if mode in ('delta', 'log_delta'):
                y_pred = pred + y_true_packed[:, 1:2]  # shape (B, 1)
                y_true = y_true_packed[:, 0]

            elif mode in ['mse', 'mae', 'mape', 'log']:
                y_pred = pred
                y_true = y_true_packed[:, 0]
            else:
                raise ValueError(f"Unsupported mode in evaluation: {mode}")

            for i, v in enumerate(self.sparse_features):
                if v == self.hour_flag:
                    hour_idx = tf.cast(tf.gather(inputs[self.sparse_group_name], indices=i, axis=1) - 1, tf.int64)
                    hour_idx = tf.clip_by_value(hour_idx, 0, self.num_heads - 1)

            for head in range(self.num_heads):
                idxs = tf.where(tf.equal(hour_idx, head))[:, 0]
                if tf.size(idxs) == 0:
                    continue  # 该 head 在当前 batch 没有数据，跳过

                # 选出对应 head 的 pred 和 label
                head_pred = tf.gather(y_pred, idxs)
                head_true = tf.gather(y_true, idxs)

                hour_model_pred[head] += tf.reduce_sum(head_pred)
                hour_model_true[head] += tf.reduce_sum(head_true)

        return hour_model_pred, hour_model_true
