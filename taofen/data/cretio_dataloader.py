import platform
from dataconfig import *
import sys
sys.path.append("../utils")
sys.path.append("../data/")
from utils import *
import tensorflow as tf
from tensorflow import keras

def get_cretio_data(spase_feature_names, dense_feature_names, label_feature_names, path):
    if platform.system() != 'Windows':
        dataset = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord(f"{path}/train_criteo_5w_rows.tfrecord", 512)
        eval_data = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord(f"{path}/test_criteo_1w_rows.tfrecord", 512)
        valid_data = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord(f"{path}/valid_criteo_1w_rows.tfrecord", 512)
    else:
        dataset = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord(f"{path}/train_criteo_200w_rows.tfrecord", 512)
        eval_data = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord(f"{path}/test_criteo_20w_rows.tfrecord", 512)
        valid_data = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord(f"{path}/valid_criteo_20w_rows.tfrecord", 512)

    return dataset, valid_data, eval_data

def get_cretio_input(dense_feature_names, spase_feature_names ):
    # 定义 inputs
    inputs = {name: keras.Input(shape=(1,), name=name, dtype=tf.float32) for name in dense_feature_names}
    inputs.update({
        name: keras.Input(shape=(1,), name=name, dtype=tf.string) for name in spase_feature_names
    })

    return inputs

# 返回  每个数值特征的均值和方差
def get_dataset_statics(dataset, feature_names, read_batch = 1000):
    feature_mean_std_values = {}  # {feature: 'mean': v, 'std': v}

    i = 0
    for batch in dataset:  # 遍历100个batch
        for feature in batch:
            if feature in feature_names:
                values = tf.reshape(batch[feature], [-1])  # flatten to 1D
                batch_count = tf.cast(tf.size(values), tf.float32)
                batch_mean = tf.reduce_mean(values)
                batch_var = tf.math.reduce_variance(values)  # 无偏或有偏不影响推导逻辑
                batch_M2 = batch_var * batch_count  # M2 是 方差 × n

                # 初始化
                if feature not in feature_mean_std_values:
                    feature_mean_std_values[feature] = {
                        'mean': batch_mean.numpy(),
                        'M2': batch_M2.numpy(),
                        'count': batch_count.numpy()
                    }
                else:
                    # 增量更新
                    prev = feature_mean_std_values[feature]
                    delta = batch_mean.numpy() - prev['mean']
                    total_count = prev['count'] + batch_count.numpy()

                    # 更新 mean
                    new_mean = prev['mean'] + delta * (batch_count.numpy() / total_count)

                    # 更新 M2（方差的未归一化形式）
                    new_M2 = (
                            prev['M2']
                            + batch_M2.numpy()
                            + delta ** 2 * prev['count'] * batch_count.numpy() / total_count
                    )

                    # 保存更新结果
                    feature_mean_std_values[feature] = {
                        'mean': new_mean,
                        'M2': new_M2,
                        'count': total_count
                    }
        i += 1
        if i >= read_batch:
            break
    # 最后汇总结果
    res = {}
    for feature in feature_mean_std_values:
        res[feature] = {}
        values = feature_mean_std_values[feature]
        std = (values['M2'] / values['count']) ** 0.5
        res[feature]['mean'] = values['mean']
        res[feature]['std'] = std
    print("...读取统计特征over...")
    return res





