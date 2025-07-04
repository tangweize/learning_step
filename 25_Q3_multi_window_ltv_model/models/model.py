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



class Dense_Process_LOG_NORMAL_Layer(tf.keras.layers.Layer):
    def __init__(self, dense_cnt_features, dense_price_features, dense_duration_features):
        super().__init__()
        self.dense_cnt_features = dense_cnt_features
        self.dense_price_features = dense_price_features
        self.dense_duration_features = dense_duration_features
        self.concat_layer = tf.keras.layers.Concatenate()
        self.bn_layers = {}  # 字段 -> BN层

    def build(self, input_shape):
        for field in self.dense_cnt_features + self.dense_price_features + self.dense_duration_features:
            self.bn_layers[field] = tf.keras.layers.BatchNormalization(name=f"{field}_bn")

    def call(self, inputs, training=False):
        processed_dense_features = []
        for field, input_tensor in inputs.items():
            x = tf.cast(input_tensor, tf.float32)
            x = tf.maximum(x, 0.0)

            if field in self.dense_cnt_features:
                x = tf.math.log1p(x + 1) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
            elif field in self.dense_price_features or field in self.dense_duration_features:
                x = tf.math.log1p(x + 1) / tf.math.log(tf.constant(10.0, dtype=tf.float32))

            x = self.bn_layers[field](x, training=training)  # 注意加 training 标志
            processed_dense_features.append(x)

        return self.concat_layer(processed_dense_features)


# 不做log处理，只加归一化。
class Dense_NO_Process_Layer(layers.Layer):
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

                processed_dense_features.append(input_tensor)
            elif field in self.dense_price_features or field in self.dense_duration_features:
                input_tensor = tf.cast(input_tensor, tf.float32)  # 显式转 float
                processed_dense_features.append(input_tensor)
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


from tensorflow.keras import layers, Sequential
class DNN(layers.Layer):
    def __init__(self, units, activation='relu', use_bn=False, dropout_rate=0.0):
        """
        :param units: List[int]，每层的神经元数量
        :param activation: 激活函数，比如 'relu'
        :param use_bn: 是否使用 BatchNormalization
        :param dropout_rate: Dropout 比例（0 表示不使用）
        """
        super().__init__()
        self.model = Sequential()
        for i, unit in enumerate(units):
            self.model.add(layers.Dense(unit, activation=None))  # activation 单独加，避免 BN 影响
            if use_bn:
                self.model.add(layers.BatchNormalization())
            self.model.add(layers.Activation(activation))
            if dropout_rate > 0.0:
                self.model.add(layers.Dropout(dropout_rate))

    def call(self, inputs, training=None):
        return self.model(inputs, training=training)


# 正式模型
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
                 Dense_feature_Layer = Dense_Process_LOG_Layer,
                 Sparse_feature_Layer = Sparse_Process_Layer,
                 hour_flag = 'request_hour_diff',
                 dnn_param = {"use_bn":False, "drop_out":0.0 }
                 ):
        super().__init__()
        self.num_heads = num_heads


        self.process_dense_layer = Dense_feature_Layer(dense_cnt_feature_name, dense_price_feature_name, dense_duration_feature_name)

        # 配置dnn 的参数， 是否加bn， 是否 加 dropout
        self.dnn_params = dnn_param
        is_use_bn = self.dnn_params["use_bn"]
        dropout_rate = self.dnn_params["drop_out"]


        self.process_emb_layer = Sparse_feature_Layer(sparse_group_name, user_sparse_features, emb_features)
        self.sparse_features = user_sparse_features
        self.sparse_group_name = sparse_group_name
        self.hour_flag = hour_flag


        self.sharebottom = DNN(units, use_bn = is_use_bn, dropout_rate = dropout_rate )
        self.hour2headnn = [ HEAD_DNN(head_units) for i in range(self.num_heads)]
        self.concat_layer = layers.Concatenate()
        self.dense_bn = tf.keras.layers.BatchNormalization()


    def call(self, inputs, training=False):
        dense_tensor = self.process_dense_layer(inputs)
        dense_tensor = self.dense_bn(dense_tensor, training=training)
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
            # tf.print("loss:",losses)
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

            if mode in ('delta', 'log_delta', 'delta_tweedie', 'mse_regular') :
                y_pred = pred + tf.reshape(y_true_packed[:, 1], (-1, 1))  # shape (B, 1)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))

            elif mode in ['mse', 'mae', 'mape', 'log', 'tweedie']:
                y_pred = pred
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))
            elif mode in ['binary']:
                y_pred = tf.sigmoid(pred)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))


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
        def safe_concat_and_maybe_squeeze(tensor_list):
            out = tf.concat(tensor_list, axis=0)
            if len(out.shape) == 2 and out.shape[1] == 1:
                out = tf.squeeze(out, axis=1)
            return out

        for head in range(self.num_heads):
            if len(hour_model_pred[head]) == 0:
                continue

            hour_model_pred[head] = safe_concat_and_maybe_squeeze(hour_model_pred[head])
            hour_model_true[head] = safe_concat_and_maybe_squeeze(hour_model_true[head])
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




    def evaluate_rank(self, test_dataset, is_plot = True ):

        head_pred, head_true = self.predict_head_score(test_dataset)

        rank_res = {}
        for head in head_pred.keys():
            head_name = f"{head + 1}_h rank_score"
            pred = head_pred[head]
            true = head_true[head]
            if len(pred):
                rank_res[head_name] = round(self.calculate_area_under_gain_curve(pred, true, head_name, is_plot), 4)

        return rank_res

    def calculate_area_under_gain_curve(self, pred_list, true_list, head_name="", is_plot = True):
        # 将零维张量列表转换为一维 NumPy 数组
        pred = pred_list.numpy()
        true = true_list.numpy()

        # 创建 DataFrame
        df = pd.DataFrame({'pred': pred, 'true': true})

        # 【1】预测值排序的增益曲线
        df_pred_sorted = df.sort_values(by='pred', ascending=False).copy()
        df_pred_sorted['cumulative_percentage_customers'] = np.arange(1, len(df_pred_sorted) + 1) / len(df_pred_sorted)
        df_pred_sorted['cumulative_percentage_ltv'] = df_pred_sorted['true'].cumsum() / df_pred_sorted['true'].sum()
        area_pred = np.trapz(df_pred_sorted['cumulative_percentage_ltv'],
                             df_pred_sorted['cumulative_percentage_customers'])

        if is_plot:
            # 【2】真实值排序的理想增益曲线（Ground Truth 理想线）
            df_true_sorted = df.sort_values(by='true', ascending=False).copy()
            df_true_sorted['cumulative_percentage_customers'] = np.arange(1, len(df_true_sorted) + 1) / len(df_true_sorted)
            df_true_sorted['cumulative_percentage_ltv'] = df_true_sorted['true'].cumsum() / df_true_sorted['true'].sum()
            area_true = np.trapz(df_true_sorted['cumulative_percentage_ltv'],
                                 df_true_sorted['cumulative_percentage_customers'])

            # 【3】绘图
            plt.figure(figsize=(10, 6))
            plt.plot(df_pred_sorted['cumulative_percentage_customers'],
                     df_pred_sorted['cumulative_percentage_ltv'],
                     label="Gain Curve (Predicted)", linewidth=2)
            plt.plot(df_true_sorted['cumulative_percentage_customers'],
                     df_true_sorted['cumulative_percentage_ltv'],
                     label="Ideal Gain Curve (Ground Truth Sorted)",
                     linestyle='--', color='black', linewidth=2)
            plt.plot([0, 1], [0, 1], linestyle=':', color='gray', label="Random Model")

            plt.xlabel('Cumulative Percentage of Customers')
            plt.ylabel('Cumulative Percentage of Total LTV')
            plt.title(f'{head_name} Gain Chart')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return area_pred


    def evaluate_budget_exp(self, test_dataset, is_plot = True ):

        head_pred, head_true = self.predict_head_score(test_dataset)

        rank_res = {}
        for head in head_pred.keys():
            head_name = f"{head + 1}_h rank_score"
            pred = head_pred[head]
            true = head_true[head]
            if len(pred):
                self.evaluate_ltv_bucket_chart(pred, true, head_name)

        return -1


    def evaluate_ltv_bucket_chart(self, pred_list, true_list, head_name=""):
        # 将零维张量列表转换为一维 NumPy 数组
        pred = pred_list.numpy()
        true = true_list.numpy()

        # 创建 DataFrame
        df = pd.DataFrame({'pred': pred, 'true': true})

        # 分成10分位桶（基于预测值）
        df['bucket'] = pd.qcut(df['pred'], q=10, labels=False, duplicates='drop')

        # 每桶计算真实值和预测值的均值
        bucket_stats = df.groupby('bucket').agg({
            'true': 'mean',
            'pred': 'mean'
        }).rename(columns={'true': 'True LTV', 'pred': 'Predicted LTV'})

        # 可视化
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 6))
        x = np.arange(len(bucket_stats))
        width = 0.35

        plt.bar(x - width/2, bucket_stats['True LTV'], width, label='True LTV')
        plt.bar(x + width/2, bucket_stats['Predicted LTV'], width, label='Predicted LTV')

        plt.xlabel('Decile Buckets (by Prediction)')
        plt.ylabel('Mean LTV')
        plt.title(f'LTV Decile Chart - {head_name}')
        plt.xticks(x, [f'Q{i+1}' for i in x])
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()



# MMOE
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

            if mode in ('delta', 'log_delta', 'delta_tweedie', 'mse_regular') :
                y_pred = pred + tf.reshape(y_true_packed[:, 1], (-1, 1))  # shape (B, 1)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))

            elif mode in ['mse', 'mae', 'mape', 'log']:
                y_pred = pred
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))
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

"""
demo model 之前用keras 不做任何处理效果都不差。
将复现一下，寻找其中diff
"""

class NO_Process_Layer(layers.Layer):
    def __init__(self, dense_cnt_features, dense_price_features, dense_duration_features, user_sparse_features):
        super().__init__()

        self.features = [dense_cnt_features
                        ,dense_price_features
                        , dense_duration_features
                        ,user_sparse_features
                         ]
        self.concat_layer = layers.Concatenate()
    def call(self, inputs):
        processed_dense_features = []
        for field, input_tensor in inputs.items():
            if field in self.features:
                input_tensor = tf.cast(input_tensor, tf.float32)  # 显式转 float
                processed_dense_features.append(input_tensor)
        return self.concat_layer(processed_dense_features)


class Single_HEAD_LTV_MODEL(keras.Model):
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
                 hour_flag = 'request_hour_diff',
                 is_log = True):
        super().__init__()
        self.num_heads = num_heads

        self.process_dense_layer = NO_Process_Layer(dense_cnt_feature_name, dense_price_feature_name, dense_duration_feature_name, sparse_group_name)

        self.sparse_features = user_sparse_features
        self.sparse_group_name = sparse_group_name
        self.hour_flag = hour_flag

        self.sharebottom = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1)
        ])

        # Each head as a Sequential model
        # self.hour2headnn = [ tf.keras.layers.Dense(1) for _ in range(self.num_heads)
        # ]
        self.concat_layer = layers.Concatenate()
    def call(self, inputs, training=False):
        input_tensor = self.process_dense_layer(inputs)
        print(input_tensor.shape)
        sharebottom_out = self.sharebottom(input_tensor)

        # outputs = [head(sharebottom_out) for head in self.hour2headnn]


        # hour_idx = -1
        # for i, v in enumerate(self.sparse_features):
        #     if v == self.hour_flag:
        #         hour_idx = tf.cast(tf.gather(inputs[self.sparse_group_name], indices=i, axis = 1) - 1, tf.int64)
        #         hour_idx = tf.clip_by_value(hour_idx, 0, self.num_heads - 1)

        # outputs = self.concat_layer(outputs)
        # selected = tf.gather(outputs, hour_idx, axis=1, batch_dims=1)
        # selected = tf.expand_dims(selected, axis=1)  # 模拟 keep_dim=True 的效果


        return sharebottom_out

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

            if mode in ('delta', 'log_delta', 'delta_tweedie', 'mse_regular') :
                y_pred = pred + tf.reshape(y_true_packed[:, 1], (-1, 1))  # shape (B, 1)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))

            elif mode in ['mse', 'mae', 'mape', 'log']:
                y_pred = pred
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))
            elif mode in ['binary']:
                y_pred = tf.sigmoid(pred)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))


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
        def safe_concat_and_maybe_squeeze(tensor_list):
            out = tf.concat(tensor_list, axis=0)
            if len(out.shape) == 2 and out.shape[1] == 1:
                out = tf.squeeze(out, axis=1)
            return out

        for head in range(self.num_heads):
            if len(hour_model_pred[head]) == 0:
                continue

            hour_model_pred[head] = safe_concat_and_maybe_squeeze(hour_model_pred[head])
            hour_model_true[head] = safe_concat_and_maybe_squeeze(hour_model_true[head])
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
            if len(pred):
                rank_res[head_name] = self.calculate_area_under_gain_curve(pred, true, head_name)

        return rank_res

    def calculate_area_under_gain_curve(self, pred_list, true_list, head_name=""):
        # 将零维张量列表转换为一维 NumPy 数组
        pred = pred_list.numpy()
        true = true_list.numpy()

        # 创建 DataFrame
        df = pd.DataFrame({'pred': pred, 'true': true})

        # 【1】预测值排序的增益曲线
        df_pred_sorted = df.sort_values(by='pred', ascending=False).copy()
        df_pred_sorted['cumulative_percentage_customers'] = np.arange(1, len(df_pred_sorted) + 1) / len(df_pred_sorted)
        df_pred_sorted['cumulative_percentage_ltv'] = df_pred_sorted['true'].cumsum() / df_pred_sorted['true'].sum()
        area_pred = np.trapz(df_pred_sorted['cumulative_percentage_ltv'],
                             df_pred_sorted['cumulative_percentage_customers'])

        # 【2】真实值排序的理想增益曲线（Ground Truth 理想线）
        df_true_sorted = df.sort_values(by='true', ascending=False).copy()
        df_true_sorted['cumulative_percentage_customers'] = np.arange(1, len(df_true_sorted) + 1) / len(df_true_sorted)
        df_true_sorted['cumulative_percentage_ltv'] = df_true_sorted['true'].cumsum() / df_true_sorted['true'].sum()
        area_true = np.trapz(df_true_sorted['cumulative_percentage_ltv'],
                             df_true_sorted['cumulative_percentage_customers'])

        # 【3】绘图
        plt.figure(figsize=(10, 6))
        plt.plot(df_pred_sorted['cumulative_percentage_customers'],
                 df_pred_sorted['cumulative_percentage_ltv'],
                 label="Gain Curve (Predicted)", linewidth=2)
        plt.plot(df_true_sorted['cumulative_percentage_customers'],
                 df_true_sorted['cumulative_percentage_ltv'],
                 label="Ideal Gain Curve (Ground Truth Sorted)",
                 linestyle='--', color='black', linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle=':', color='gray', label="Random Model")

        plt.xlabel('Cumulative Percentage of Customers')
        plt.ylabel('Cumulative Percentage of Total LTV')
        plt.title(f'{head_name} Gain Chart')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return area_pred


class MULTI_HEAD_LTV_MODEL_ADD_HEAD_BN(keras.Model):
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
                 Dense_feature_Layer = Dense_Process_LOG_Layer,
                 Sparse_feature_Layer = Sparse_Process_Layer,
                 hour_flag = 'request_hour_diff',
                 dnn_param = {"use_bn":False, "drop_out":0.0 }
                 ):
        super().__init__()
        self.num_heads = num_heads


        self.process_dense_layer = Dense_feature_Layer(dense_cnt_feature_name, dense_price_feature_name, dense_duration_feature_name)

        # 配置dnn 的参数， 是否加bn， 是否 加 dropout
        self.dnn_params = dnn_param
        is_use_bn = self.dnn_params["use_bn"]
        dropout_rate = self.dnn_params["drop_out"]


        self.process_emb_layer = Sparse_feature_Layer(sparse_group_name, user_sparse_features, emb_features)
        self.sparse_features = user_sparse_features
        self.sparse_group_name = sparse_group_name
        self.hour_flag = hour_flag


        self.sharebottom = DNN(units, use_bn = is_use_bn, dropout_rate = dropout_rate )

        self.hour2headnn = [ HEAD_DNN(head_units) for i in range(self.num_heads)]
        self.hour2headbn = [tf.keras.layers.BatchNormalization() for i in range(self.num_heads)]

        self.concat_layer = layers.Concatenate()
        self.dense_bn = tf.keras.layers.BatchNormalization()


    def call(self, inputs, training=False):
        dense_tensor = self.process_dense_layer(inputs)
        dense_tensor = self.dense_bn(dense_tensor, training=training)
        emb_tesor = self.process_emb_layer(inputs)

        input_tensor = layers.Concatenate()([dense_tensor, emb_tesor])

        sharebottom_out = self.sharebottom(input_tensor)

        outputs = [headnn(headbn(sharebottom_out)) for headnn, headbn in zip(self.hour2headnn, self.hour2headbn)]

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
            # tf.print("loss:",losses)
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

            if mode in ('delta', 'log_delta', 'delta_tweedie', 'mse_regular') :
                y_pred = pred + tf.reshape(y_true_packed[:, 1], (-1, 1))  # shape (B, 1)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))

            elif mode in ['mse', 'mae', 'mape', 'log', 'tweedie']:
                y_pred = pred
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))
            elif mode in ['binary']:
                y_pred = tf.sigmoid(pred)
                y_true = tf.reshape(y_true_packed[:, 0], (-1, 1))


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
        def safe_concat_and_maybe_squeeze(tensor_list):
            out = tf.concat(tensor_list, axis=0)
            if len(out.shape) == 2 and out.shape[1] == 1:
                out = tf.squeeze(out, axis=1)
            return out

        for head in range(self.num_heads):
            if len(hour_model_pred[head]) == 0:
                continue

            hour_model_pred[head] = safe_concat_and_maybe_squeeze(hour_model_pred[head])
            hour_model_true[head] = safe_concat_and_maybe_squeeze(hour_model_true[head])
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




    def evaluate_rank(self, test_dataset, is_plot = True ):

        head_pred, head_true = self.predict_head_score(test_dataset)

        rank_res = {}
        for head in head_pred.keys():
            head_name = f"{head + 1}_h rank_score"
            pred = head_pred[head]
            true = head_true[head]
            if len(pred):
                rank_res[head_name] = round(self.calculate_area_under_gain_curve(pred, true, head_name, is_plot), 4)

        return rank_res

    def calculate_area_under_gain_curve(self, pred_list, true_list, head_name="", is_plot = True):
        # 将零维张量列表转换为一维 NumPy 数组
        pred = pred_list.numpy()
        true = true_list.numpy()

        # 创建 DataFrame
        df = pd.DataFrame({'pred': pred, 'true': true})

        # 【1】预测值排序的增益曲线
        df_pred_sorted = df.sort_values(by='pred', ascending=False).copy()
        df_pred_sorted['cumulative_percentage_customers'] = np.arange(1, len(df_pred_sorted) + 1) / len(df_pred_sorted)
        df_pred_sorted['cumulative_percentage_ltv'] = df_pred_sorted['true'].cumsum() / df_pred_sorted['true'].sum()
        area_pred = np.trapz(df_pred_sorted['cumulative_percentage_ltv'],
                             df_pred_sorted['cumulative_percentage_customers'])

        if is_plot:
            # 【2】真实值排序的理想增益曲线（Ground Truth 理想线）
            df_true_sorted = df.sort_values(by='true', ascending=False).copy()
            df_true_sorted['cumulative_percentage_customers'] = np.arange(1, len(df_true_sorted) + 1) / len(df_true_sorted)
            df_true_sorted['cumulative_percentage_ltv'] = df_true_sorted['true'].cumsum() / df_true_sorted['true'].sum()
            area_true = np.trapz(df_true_sorted['cumulative_percentage_ltv'],
                                 df_true_sorted['cumulative_percentage_customers'])

            # 【3】绘图
            plt.figure(figsize=(10, 6))
            plt.plot(df_pred_sorted['cumulative_percentage_customers'],
                     df_pred_sorted['cumulative_percentage_ltv'],
                     label="Gain Curve (Predicted)", linewidth=2)
            plt.plot(df_true_sorted['cumulative_percentage_customers'],
                     df_true_sorted['cumulative_percentage_ltv'],
                     label="Ideal Gain Curve (Ground Truth Sorted)",
                     linestyle='--', color='black', linewidth=2)
            plt.plot([0, 1], [0, 1], linestyle=':', color='gray', label="Random Model")

            plt.xlabel('Cumulative Percentage of Customers')
            plt.ylabel('Cumulative Percentage of Total LTV')
            plt.title(f'{head_name} Gain Chart')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return area_pred


    def evaluate_budget_exp(self, test_dataset, is_plot = True ):

        head_pred, head_true = self.predict_head_score(test_dataset)

        rank_res = {}
        for head in head_pred.keys():
            head_name = f"{head + 1}_h rank_score"
            pred = head_pred[head]
            true = head_true[head]
            if len(pred):
                self.evaluate_ltv_bucket_chart(pred, true, head_name)

        return -1


    def evaluate_ltv_bucket_chart(self, pred_list, true_list, head_name=""):
        # 将零维张量列表转换为一维 NumPy 数组
        pred = pred_list.numpy()
        true = true_list.numpy()

        # 创建 DataFrame
        df = pd.DataFrame({'pred': pred, 'true': true})

        # 分成10分位桶（基于预测值）
        df['bucket'] = pd.qcut(df['pred'], q=10, labels=False, duplicates='drop')

        # 每桶计算真实值和预测值的均值
        bucket_stats = df.groupby('bucket').agg({
            'true': 'mean',
            'pred': 'mean'
        }).rename(columns={'true': 'True LTV', 'pred': 'Predicted LTV'})

        # 可视化
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 6))
        x = np.arange(len(bucket_stats))
        width = 0.35

        plt.bar(x - width/2, bucket_stats['True LTV'], width, label='True LTV')
        plt.bar(x + width/2, bucket_stats['Predicted LTV'], width, label='Predicted LTV')

        plt.xlabel('Decile Buckets (by Prediction)')
        plt.ylabel('Mean LTV')
        plt.title(f'LTV Decile Chart - {head_name}')
        plt.xticks(x, [f'Q{i+1}' for i in x])
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
