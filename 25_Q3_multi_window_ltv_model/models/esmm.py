import os
import tf2onnx
import shutil
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Multiply
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model, load_model


# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F 
# from pyspark.sql.types import FloatType

# 解析单个样本的函数
def parse_example(serialized_example, feature_description, feature_cols):
    parsed_features = tf.io.parse_single_example(serialized_example, feature_description)
    for col in feature_cols:
        dense_vector_values = tf.sparse.to_dense(parsed_features[col], default_value=0.0)
        parsed_features[f'{col}_values'] = dense_vector_values  # 转换为密集张量
    return parsed_features


# 提取训练集和验证集
def is_deviceid_train(data):
    # 获取 deviceid 的最后一个字符
    last_char = tf.strings.substr(data['deviceid'], -1, 1)
    # 判断是否是 0-9或者a/b
    return tf.strings.regex_full_match(last_char, "[0-9ab]")


def is_deviceid_valid(data):
    # 获取 deviceid 的最后一个字符
    last_char = tf.strings.substr(data['deviceid'], -1, 1)
    # 判断是否是 c-f
    return tf.strings.regex_full_match(last_char, "[c-f]")


def extract_feature(data, feature_dict, feature_col, target_dtype=tf.float32):
    """
    提取指定特征，将其转换为目标数据类型，并堆叠为张量。

    参数:
    - data: tf.data.Dataset，输入数据集。
    - feature_dict: dict，包含特征名称及索引的字典。
    - target_dtype: 数据类型，例如 tf.float32 或 tf.int32。

    返回:
    - tf.data.Dataset，映射后的数据集，输出形状为 (batch_size, num_features) 的张量。
    """

    def map_fn(x):
        # 将每个特征转换为目标数据类型并存入列表
        tensors = []
        for col in feature_dict.keys():
            t = tf.cast(x[feature_col][feature_dict[col]["index"]], target_dtype)
            tensors.append(t)
        # 将所有特征堆叠成一个张量
        return tf.stack(tensors, axis=-1)

    # 使用 map 处理数据集
    return data.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)


# 提取目标值
def extract_target(data, target):
    return data.map(lambda x: x[target])


# 汇总数据处理并构建 tf.data.Dataset 格式数据集
def data_process_pipeline(parsed_dataset, stage,
                          ctr_user_num_feature_dict, ctr_item_num_feature_dict,
                          cvr_user_num_feature_dict, cvr_item_num_feature_dict,
                          num_dtype,
                          user_cat_feature_dict, item_cat_feature_dict, cat_dtype):
    if stage == 'train':
        dataset = parsed_dataset.filter(is_deviceid_train)
    elif stage == 'valid':
        dataset = parsed_dataset.filter(is_deviceid_valid)
    elif stage == 'test':
        dataset = parsed_dataset
    else:
        raise ValueError("Invalid stage")

    ctr_user_num_feature = extract_feature(dataset, ctr_user_num_feature_dict, 'ctr_user_num_features_values',
                                           num_dtype)
    ctr_item_num_feature = extract_feature(dataset, ctr_item_num_feature_dict, 'ctr_item_num_features_values',
                                           num_dtype)

    cvr_user_num_feature = extract_feature(dataset, cvr_user_num_feature_dict, 'cvr_user_num_features_values',
                                           num_dtype)
    cvr_item_num_feature = extract_feature(dataset, cvr_item_num_feature_dict, 'cvr_item_num_features_values',
                                           num_dtype)

    user_cat_feature = extract_feature(dataset, user_cat_feature_dict, 'user_cat_features_values', cat_dtype)
    item_cat_feature = extract_feature(dataset, item_cat_feature_dict, 'item_cat_features_values', cat_dtype)

    ctr_target = extract_target(dataset, 'is_click')
    cvr_target = extract_target(dataset, 'is_pay')

    return tf.data.Dataset.zip((
        (
            ctr_user_num_feature, ctr_item_num_feature, cvr_user_num_feature, cvr_item_num_feature,
            user_cat_feature, item_cat_feature
        ),
        (ctr_target, cvr_target)
    ))


def extract_single_feature(sample, feature_dict, feature_col, dtype):
    indices = [v['index'] for v in feature_dict.values()]
    # 快速提取需要的特征值
    values = tf.gather(sample[feature_col], indices)
    return tf.cast(values, dtype)


# 汇总数据处理并构建 tf.data.Dataset 格式数据集
def data_process_pipeline_v2(parsed_dataset, stage,
                             ctr_user_num_feature_dict, ctr_item_num_feature_dict,
                             cvr_user_num_feature_dict, cvr_item_num_feature_dict,
                             num_dtype,
                             user_cat_feature_dict, item_cat_feature_dict, cat_dtype):
    if stage == 'train':
        dataset = parsed_dataset.filter(is_deviceid_train)
    elif stage == 'valid':
        dataset = parsed_dataset.filter(is_deviceid_valid)
    elif stage == 'test':
        dataset = parsed_dataset
    else:
        raise ValueError("Invalid stage")

    # 定义统一处理的 map 函数
    def extract_all_features(data):
        # 一次性提取所有数值特征
        ctr_user_num = extract_single_feature(data, ctr_user_num_feature_dict, 'ctr_user_num_features_values',
                                              num_dtype)
        ctr_item_num = extract_single_feature(data, ctr_item_num_feature_dict, 'ctr_item_num_features_values',
                                              num_dtype)
        cvr_user_num = extract_single_feature(data, cvr_user_num_feature_dict, 'cvr_user_num_features_values',
                                              num_dtype)
        cvr_item_num = extract_single_feature(data, cvr_item_num_feature_dict, 'cvr_item_num_features_values',
                                              num_dtype)
        # 提取分类特征
        user_cat = extract_single_feature(data, user_cat_feature_dict, 'user_cat_features_values', cat_dtype)
        item_cat = extract_single_feature(data, item_cat_feature_dict, 'item_cat_features_values', cat_dtype)
        # 返回整合特征和标签
        return (
            (ctr_user_num, ctr_item_num, cvr_user_num, cvr_item_num, user_cat, item_cat),
            (data['is_click'], data['is_pay'])
        )

    # 合并后的统一map处理
    processed_dataset = dataset.map(extract_all_features, num_parallel_calls=tf.data.AUTOTUNE)

    return processed_dataset


# 使用多线程加速从tf.dataset转换到numpy array
def dataset2np(dataset):
    start_time = time()
    optimized_dataset = dataset.map(
        lambda x: x,
        num_parallel_calls=tf.data.experimental.AUTOTUNE  # 自动调节线程数
    )

    # 批量处理 + 转 NumPy
    batched_dataset = optimized_dataset.batch(8192)
    data_np = np.concatenate([batch.numpy() for batch in batched_dataset])
    print(f'Running time is: {time() - start_time} seconds.')
    return data_np


# 定义 AUC 的 UDF
def calculate_auc(y_true, y_pred):
    if len(set(y_true)) < 2:  # 确保有正负样本
        return None
    return float(roc_auc_score(y_true, y_pred))


def auc_calculation(y_true, y_predict_dataset, deviceid_np):
    # 计算整体AUC
    if isinstance(y_true, DatasetV2):
        y_true_np = dataset2np(y_true)
    else:
        y_true_np = y_true
    # y_predict_np = y_predict_dataset.numpy().flatten()
    y_predict_np = y_predict_dataset.flatten()

    overall_auc = roc_auc_score(y_true_np, y_predict_np)

    # 计算deviceid gauc
    test_df = pd.DataFrame({
        'device_id': deviceid_np,
        'y_true': y_true_np,
        'y_pred': y_predict_np
    })
    test_sdf = spark.createDataFrame(test_df).repartition(200, "device_id")  # 根据设备 ID 分区

    auc_udf = F.udf(calculate_auc, FloatType())

    # 按设备分组计算 AUC
    group_auc = test_sdf.groupBy("device_id").agg(
        F.collect_list("y_true").alias("y_true_list"),
        F.collect_list("y_pred").alias("y_pred_list")
    ).withColumn(
        "auc", auc_udf(F.col("y_true_list"), F.col("y_pred_list"))
    ).select("device_id", "auc")

    device_counts = test_sdf.groupBy("device_id").agg(
        F.count("*").alias("count")
    )
    # 合并 AUC 和样本数量
    group_stats = group_auc.join(device_counts, on="device_id")
    # 过滤掉只有单一标签的分组
    group_stats = group_stats.filter(F.col("auc").isNotNull())

    # 计算加权 GAUC
    weighted_gauc = group_stats.select(
        (F.sum(F.col("auc") * F.col("count")) / F.sum("count")).alias("weighted_gauc")
    ).collect()[0]["weighted_gauc"]

    return overall_auc, weighted_gauc


# 定义一阶polyloss
def poly1_binary_crossentropy(y_true, y_pred, epsilon=1.0):
    """
    Poly-1 loss = CE + epsilon * (1 - pt)
    y_true: [batch_size, 1]
    y_pred: [batch_size, 1], sigmoid/logit之后的概率
    epsilon: float, 惩罚项的系数
    """
    # 保证 y_true 和 y_pred 的 dtype 一致
    y_true = tf.cast(y_true, tf.float32)

    # 计算常规的 binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # pt = p(y_pred正确) = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # Poly-1 惩罚项
    poly1_term = epsilon * tf.pow((1 - pt), 1.0)

    return bce + poly1_term


# def model_transfer_and_save(model_file):
#     model_path = '/root/hdfs/write/models/model'
#     tmp_file = '/tf/model.onnx'
#     if (not os.path.exists(model_path)):
#         os.makedirs(model_path)
#     model = load_model(model_file)
#     # 转换为 ONNX
#     onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
#     with open(tmp_file, "wb") as f:
#         f.write(onnx_model.SerializeToString())
#     os.system("cp /tf/model.onnx /root/hdfs/write/models/model/")
#     os.system("ls -al /root/hdfs/write/models/model")
#     os.system("chmod -R 777 /root/hdfs/write/")

def model_transfer_and_save(model_file, trainset_duration, feat_version):
    """
    将 Keras 模型转换为 ONNX 格式并保存到指定路径。

    Args:
        model_file (str): 输入 Keras 模型文件路径（.keras 或其他支持的格式）。

    Raises:
        FileNotFoundError: 如果输入模型文件不存在。
        Exception: 如果 ONNX 转换失败。
    """
    model_path = '/root/hdfs/write/models/model'
    tmp_file_onnx = '/tf/model.onnx'
    tmp_file_keras = f"/tf/model_{trainset_duration}_feat{feat_version}.keras"

    # 检查输入模型文件是否存在
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"模型文件 {model_file} 不存在，请检查路径！")

    # 创建保存路径（如果不存在）
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    try:
        # 加载 Keras 模型
        print(f"加载模型文件：{model_file}")
        model = load_model(model_file)

        # 转换为 ONNX 格式
        print("正在将模型转换为 ONNX 格式...")
        onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

        # 保存 ONNX 模型到临时文件
        print(f"保存临时 ONNX 文件到：{tmp_file_onnx}")
        with open(tmp_file_onnx, "wb") as f:
            f.write(onnx_model.SerializeToString())
        # 保存 keras 模型到临时文件
        print(f"保存临时 keras 文件到：{tmp_file_keras}")
        model.save(tmp_file_keras)

        # 使用 shutil 移动文件到目标路径
        target_file_onnx = os.path.join(model_path, "model.onnx")
        print(f"正在移动onnx文件到目标路径：{target_file_onnx}")
        shutil.move(tmp_file_onnx, target_file_onnx)
        print(f"onnx模型已保存到：{target_file_onnx}")

        target_file_keras = os.path.join(
            model_path,
            f"model_{trainset_duration}_feat{feat_version}_{datetime.datetime.now().strftime('%Y%m%d')}.keras"
        )
        print(f"正在移动keras文件到目标路径：{target_file_keras}")
        shutil.move(tmp_file_keras, target_file_keras)
        print(f"keras模型已保存到：{target_file_keras}")

        # 检查文件是否成功复制
        print("保存路径下的文件列表：")
        os.system(f"ls -al {model_path}")

        # 修改权限
        print("修改写入路径权限...")
        os.system("chmod -R 777 /root/hdfs/write/")
        print("权限修改完成！")

    except Exception as e:
        print(f"模型转换或保存失败：{e}")
        raise


class CTCVRNet_v0:
    # 定义一个字典 self.embed 用于保存每个类别型特征的嵌入层，不论是构建 CTR 模型还是 CVR 模型，这些 Embedding 层都是共享的
    def __init__(self, cat_feautre_dict, embedding_size):
        # 为每个类别型特征创建一个嵌入层，输入大小为 v（该特征的种类数）,每个类别型特征都会对应一个独立的嵌入层
        self.embed = {
            k: layers.Embedding(v, embedding_size, dtype='float16') for k, v in cat_feautre_dict.items()
        }

    # 遍历类别型特征字典，逐个对特征嵌入；返回数值特征+嵌入后的类别特征向量
    def build_feature_embedding(self, num_input, cat_input, cat_feature_dict, embedding_size):
        embeddings = [
            layers.Reshape((embedding_size,))(self.embed[feature_name](cat_input[:, idx]))
            for idx, feature_name in enumerate(cat_feature_dict.keys())
        ]
        return layers.concatenate([num_input] + embeddings, axis=-1)

    # 多层全连接网络抽象化
    def build_dense_layers(self, feature):
        feature = layers.Dropout(0.5)(feature)
        feature = layers.BatchNormalization()(feature)
        feature = layers.Dense(128, activation='relu')(feature)
        feature = layers.Dense(64, activation='relu')(feature)
        return feature

    # CTR 和 CVR 网络共享逻辑
    def build_model_part(self, user_num_input, user_cat_input, \
                         item_num_input, item_cat_input, \
                         user_cat_feature_dict, item_cat_feature_dict, \
                         embedding_size, output_name):
        # 获取处理后的特征向量
        user_feature = self.build_feature_embedding(user_num_input, user_cat_input, user_cat_feature_dict,
                                                    embedding_size)
        item_feature = self.build_feature_embedding(item_num_input, item_cat_input, item_cat_feature_dict,
                                                    embedding_size)

        # 用户特征和商品特征分别经过全连接网络
        user_feature = self.build_dense_layers(user_feature)
        item_feature = self.build_dense_layers(item_feature)

        # 特征汇总，加全连接网络输出
        combined_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        combined_feature = layers.Dropout(0.5)(combined_feature)
        combined_feature = layers.BatchNormalization()(combined_feature)
        combined_feature = layers.Dense(64, activation='relu')(combined_feature)

        pred = layers.Dense(1, activation='sigmoid', name=output_name)(combined_feature)
        return pred

    # 模型主逻辑
    def build(self, user_cat_feature_dict, item_cat_feature_dict, \
              ctr_user_num_feature_dict, ctr_item_num_feature_dict, \
              cvr_user_num_feature_dict, cvr_item_num_feature_dict, \
              embedding_size):
        # 设置输入数据形状
        ctr_user_num_cnt = len(ctr_user_num_feature_dict.keys())
        ctr_item_num_cnt = len(ctr_item_num_feature_dict.keys())

        cvr_user_num_cnt = len(cvr_user_num_feature_dict.keys())
        cvr_item_num_cnt = len(cvr_item_num_feature_dict.keys())

        user_cat_cnt = len(user_cat_feature_dict.keys())
        item_cat_cnt = len(item_cat_feature_dict.keys())

        # 设置输入层
        ctr_user_num_input = layers.Input(shape=(ctr_user_num_cnt,), name='ctr_user_num_features')
        ctr_item_num_input = layers.Input(shape=(ctr_item_num_cnt,), name='ctr_item_num_features')

        cvr_user_num_input = layers.Input(shape=(cvr_user_num_cnt,), name='cvr_user_num_features')
        cvr_item_num_input = layers.Input(shape=(cvr_item_num_cnt,), name='cvr_item_num_features')

        user_cat_input = layers.Input(shape=(user_cat_cnt,), name='user_cat_features')
        item_cat_input = layers.Input(shape=(item_cat_cnt,), name='item_cat_features')

        # 搭建CTR和CVR网络
        ctr_pred = self.build_model_part(
            ctr_user_num_input, user_cat_input,
            ctr_item_num_input, item_cat_input,
            user_cat_feature_dict, item_cat_feature_dict,
            embedding_size, output_name='ctr_output')

        cvr_pred = self.build_model_part(
            cvr_user_num_input, user_cat_input,
            cvr_item_num_input, item_cat_input,
            user_cat_feature_dict, item_cat_feature_dict,
            embedding_size, output_name='cvr_output')

        # 相乘得CTCVR
        ctcvr_pred = Multiply()([ctr_pred, cvr_pred])

        model = Model(
            inputs=[ctr_user_num_input, ctr_item_num_input, cvr_user_num_input, cvr_item_num_input,
                    user_cat_input, item_cat_input],
            outputs=[ctr_pred, ctcvr_pred])

        return model


class CTCVRNet_v1:
    # 定义一个字典 self.embed 用于保存每个类别型特征的嵌入层，不论是构建 CTR 模型还是 CVR 模型，这些 Embedding 层都是共享的
    def __init__(self, cat_feautre_dict, embedding_size):
        # 为每个类别型特征创建一个嵌入层，输入大小为 v（该特征的种类数）,每个类别型特征都会对应一个独立的嵌入层
        self.embed = {
            k: layers.Embedding(v, embedding_size, dtype='float16') for k, v in cat_feautre_dict.items()
        }

    # 遍历类别型特征字典，逐个对特征嵌入；返回数值特征+嵌入后的类别特征向量
    def build_feature_embedding(self, num_input, cat_input, cat_feature_dict, embedding_size):
        embeddings = [
            layers.Reshape((embedding_size,))(self.embed[feature_name](cat_input[:, idx]))
            for idx, feature_name in enumerate(cat_feature_dict.keys())
        ]
        return layers.concatenate([num_input] + embeddings, axis=-1)

    # 多层全连接网络抽象化
    def build_dense_layers(self, feature):
        feature = layers.Dropout(0.5)(feature)
        feature = layers.BatchNormalization()(feature)
        feature = layers.Dense(128, activation='relu')(feature)
        feature = layers.Dense(64, activation='relu')(feature)
        return feature

    # CTR 和 CVR 网络共享逻辑
    def build_model_part(self, user_num_input, user_cat_input, \
                         item_num_input, item_cat_input, \
                         user_cat_feature_dict, item_cat_feature_dict, \
                         embedding_size, output_name):
        # 获取处理后的特征向量
        user_feature = self.build_feature_embedding(user_num_input, user_cat_input, user_cat_feature_dict,
                                                    embedding_size)
        item_feature = self.build_feature_embedding(item_num_input, item_cat_input, item_cat_feature_dict,
                                                    embedding_size)

        # 用户特征和商品特征分别经过全连接网络
        user_feature = self.build_dense_layers(user_feature)
        item_feature = self.build_dense_layers(item_feature)

        # 特征汇总，加全连接网络输出
        combined_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        combined_feature = layers.Dropout(0.5)(combined_feature)
        combined_feature = layers.BatchNormalization()(combined_feature)
        combined_feature = layers.Dense(64, activation='relu')(combined_feature)

        pred = layers.Dense(1, activation='sigmoid', name=output_name)(combined_feature)
        return pred

    # 模型主逻辑
    def build(self, user_cat_feature_dict, item_cat_feature_dict, \
              user_num_feature_dict, item_num_feature_dict, \
              embedding_size):
        # 设置输入数据形状
        user_num_cnt = len(user_num_feature_dict.keys())
        item_num_cnt = len(item_num_feature_dict.keys())
        user_cat_cnt = len(user_cat_feature_dict.keys())
        item_cat_cnt = len(item_cat_feature_dict.keys())

        ctr_user_num_input = layers.Input(shape=(user_num_cnt,))
        ctr_user_cat_input = layers.Input(shape=(user_cat_cnt,))
        ctr_item_num_input = layers.Input(shape=(item_num_cnt,))
        ctr_item_cat_input = layers.Input(shape=(item_cat_cnt,))

        cvr_user_num_input = layers.Input(shape=(user_num_cnt,))
        cvr_user_cat_input = layers.Input(shape=(user_cat_cnt,))
        cvr_item_num_input = layers.Input(shape=(item_num_cnt,))
        cvr_item_cat_input = layers.Input(shape=(item_cat_cnt,))

        # 搭建CTR和CVR网络
        ctr_pred = self.build_model_part(
            ctr_user_num_input, ctr_user_cat_input,
            ctr_item_num_input, ctr_item_cat_input,
            user_cat_feature_dict, item_cat_feature_dict,
            embedding_size, output_name='ctr_output')

        cvr_pred = self.build_model_part(
            cvr_user_num_input, cvr_user_cat_input,
            cvr_item_num_input, cvr_item_cat_input,
            user_cat_feature_dict, item_cat_feature_dict,
            embedding_size, output_name='cvr_output')

        # 相乘得CTCVR
        ctcvr_pred = Multiply()([ctr_pred, cvr_pred])

        model = Model(
            inputs=[ctr_user_num_input, ctr_user_cat_input, ctr_item_num_input, ctr_item_cat_input,
                    cvr_user_num_input, cvr_user_cat_input, cvr_item_num_input, cvr_item_cat_input],
            outputs=[ctr_pred, ctcvr_pred])

        return model


class CTCVRNet_v2:
    # 定义一个字典 self.embed 用于保存每个类别型特征的嵌入层，不论是构建 CTR 模型还是 CVR 模型，这些 Embedding 层都是共享的
    def __init__(self, cat_feautre_dict, embedding_size):
        # 修改 Embedding 层，添加 L2 正则化
        self.embed = {
            k: layers.Embedding(
                v,
                embedding_size,
                embeddings_regularizer=regularizers.l2(0.01),  # L2 正则化
                dtype='float16'
            ) for k, v in cat_feautre_dict.items()
        }

    # 遍历类别型特征字典，逐个对特征嵌入；返回数值特征+嵌入后的类别特征向量
    def build_feature_embedding(self, num_input, cat_input, cat_feature_dict, embedding_size):
        embeddings = [
            layers.Reshape((embedding_size,))(self.embed[feature_name](cat_input[:, idx]))
            for idx, feature_name in enumerate(cat_feature_dict.keys())
        ]
        return layers.concatenate([num_input] + embeddings, axis=-1)

    # 多层全连接网络抽象化
    def build_dense_layers(self, feature):
        feature = layers.Dropout(0.5)(feature)
        feature = layers.BatchNormalization()(feature)
        # Dense 层添加 L2 正则化
        feature = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)  # L2 正则化
        )(feature)
        feature = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)  # L2 正则化
        )(feature)
        return feature

    # CTR 和 CVR 网络共享逻辑
    def build_model_part(self, user_num_input, user_cat_input, \
                         item_num_input, item_cat_input, \
                         user_cat_feature_dict, item_cat_feature_dict, \
                         embedding_size, output_name):
        # 获取处理后的特征向量
        user_feature = self.build_feature_embedding(user_num_input, user_cat_input, user_cat_feature_dict,
                                                    embedding_size)
        item_feature = self.build_feature_embedding(item_num_input, item_cat_input, item_cat_feature_dict,
                                                    embedding_size)

        # 用户特征和商品特征分别经过全连接网络
        user_feature = self.build_dense_layers(user_feature)
        item_feature = self.build_dense_layers(item_feature)

        # 特征汇总，加全连接网络输出
        combined_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        combined_feature = layers.Dropout(0.5)(combined_feature)
        combined_feature = layers.BatchNormalization()(combined_feature)
        combined_feature = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)  # L2 正则化
        )(combined_feature)

        pred = layers.Dense(1, activation='sigmoid', name=output_name)(combined_feature)
        return pred

    # 模型主逻辑
    def build(self, user_cat_feature_dict, item_cat_feature_dict, \
              ctr_user_num_feature_dict, ctr_item_num_feature_dict, \
              cvr_user_num_feature_dict, cvr_item_num_feature_dict, \
              embedding_size):
        # 设置输入数据形状
        ctr_user_num_cnt = len(ctr_user_num_feature_dict.keys())
        ctr_item_num_cnt = len(ctr_item_num_feature_dict.keys())

        cvr_user_num_cnt = len(cvr_user_num_feature_dict.keys())
        cvr_item_num_cnt = len(cvr_item_num_feature_dict.keys())

        user_cat_cnt = len(user_cat_feature_dict.keys())
        item_cat_cnt = len(item_cat_feature_dict.keys())

        # 设置输入层
        ctr_user_num_input = layers.Input(shape=(ctr_user_num_cnt,), name='ctr_user_num_features')
        ctr_item_num_input = layers.Input(shape=(ctr_item_num_cnt,), name='ctr_item_num_features')

        cvr_user_num_input = layers.Input(shape=(cvr_user_num_cnt,), name='cvr_user_num_features')
        cvr_item_num_input = layers.Input(shape=(cvr_item_num_cnt,), name='cvr_item_num_features')

        user_cat_input = layers.Input(shape=(user_cat_cnt,), name='user_cat_features')
        item_cat_input = layers.Input(shape=(item_cat_cnt,), name='item_cat_features')

        # 搭建CTR和CVR网络
        ctr_pred = self.build_model_part(
            ctr_user_num_input, user_cat_input,
            ctr_item_num_input, item_cat_input,
            user_cat_feature_dict, item_cat_feature_dict,
            embedding_size, output_name='ctr_output')

        cvr_pred = self.build_model_part(
            cvr_user_num_input, user_cat_input,
            cvr_item_num_input, item_cat_input,
            user_cat_feature_dict, item_cat_feature_dict,
            embedding_size, output_name='cvr_output')

        # 相乘得CTCVR
        ctcvr_pred = Multiply()([ctr_pred, cvr_pred])

        model = Model(
            inputs=[ctr_user_num_input, ctr_item_num_input, cvr_user_num_input, cvr_item_num_input,
                    user_cat_input, item_cat_input],
            outputs=[ctr_pred, ctcvr_pred])

        return model