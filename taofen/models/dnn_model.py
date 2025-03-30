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

    def train_step(self, train_data):

        inputs, label = train_data
        with tf.GradientTape() as tape:
            predict = self(inputs)
            # losses = self.loss(label,predict)# 报错 
            losses = tf.reduce_mean(self.loss(label, predict))  # ✅ 确保 loss 是一个数值

        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(losses, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(label, predict)
        results = {m.name: m.result() for m in self.metrics}
        return results



# 变长输入 其实是固定padding
class Input_Process_Model(Model):
    def __init__(self, inputs, behavior_list, log2_features, log10_features, sparse_features):
        super().__init__()
        numeric_features = []
        all_features = []
        for name, input in inputs.items():
            if name in log2_features:
                # x = tf.experimental.numpy.log2(input)
                x = tf.math.log1p(input) / tf.math.log(tf.constant(2.0, dtype=tf.float32))
                numeric_features.append(x)
            elif name in log10_features:
                # x = tf.experimental.numpy.log10(input)
                x = tf.math.log1p(input) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
                numeric_features.append(x)

        temp_process = layers.Concatenate()(numeric_features)

        all_features.append(temp_process)
        for name, input in inputs.items():
            if name in sparse_features:
                num_bins = 1000
                look_up = layers.Hashing(num_bins=num_bins)

                embedding = layers.Embedding(num_bins, 8, mask_zero=True)

                #
                if input.dtype != tf.int64:
                    x = look_up(input)
                else:
                    x = input

                if name in behavior_list:
                    x = tf.reduce_sum(embedding(x), axis=1)
                else:
                    x = embedding(x)
                    x = tf.squeeze(x, axis=1)

                all_features.append(x)

        feature_res = layers.Concatenate()(all_features)

        self.model = keras.Model(inputs, feature_res)

    def call(self, X):
        return self.model(X)


import tensorflow as tf


class MultiLoss_PCOC(tf.keras.losses.Loss):
    def __init__(self, lambda_pcoc=0.1, **kwargs):
        """
        lambda_pcoc: 控制 PCOC 稳定性的权重
        """
        super(MultiLoss, self).__init__(**kwargs)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.lambda_pcoc = lambda_pcoc
        self.pcoc_prev = tf.Variable(0.0, trainable=False, dtype=tf.float32)  # 记录前一 epoch 的 PCOC

    def call(self, y_true, y_pred_mask):
        y_pred, mask = y_pred_mask  # 解包 y_pred 和 mask
        return self.compute_loss(y_true, y_pred, mask)

    def compute_loss(self, y_true, y_pred, mask):
        sum_loss = 0.0
        batch_size = tf.shape(y_true)[0]
        num_classes = tf.shape(y_true)[1]
        count = tf.constant(0.0, dtype=tf.float32)

        for j in range(num_classes):
            tp_yhat = tf.expand_dims(y_pred[:, j], axis=1)
            tp_y = tf.expand_dims(y_true[:, j], axis=1)
            sample_weight = tf.expand_dims(mask[:, j], axis=1)

            sum_loss += self.bce_loss(tp_y, tp_yhat, sample_weight=sample_weight)
            count += 1

            # 计算所有类别的平均 loss
        avg_loss = sum_loss / count if count > 0 else tf.constant(0.0, dtype=tf.float32)

        # ====== 计算最后一个类别的 PCOC ======
        last_yhat = tf.expand_dims(y_pred[:, -1], axis=1)
        last_mask = tf.expand_dims(mask[:, -1], axis=1)

        # 避免除 0
        mask_sum = tf.reduce_sum(last_mask) + 1e-6
        pcoc_last = tf.reduce_sum(last_yhat * last_mask) / mask_sum

        # 计算 PCOC 平稳正则项
        reg_loss = tf.square(pcoc_last - self.pcoc_prev)

        # 更新 self.pcoc_prev 为当前 batch 的 PCOC（仅在训练时更新）
        self.pcoc_prev.assign(pcoc_last)

        # 组合 loss
        total_loss = avg_loss + self.lambda_pcoc * reg_loss

        return total_loss
