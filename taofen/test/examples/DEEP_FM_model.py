import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import sys
import datetime
sys.path.append("../../utils")
sys.path.append("../../data/")
sys.path.append("../../models/")
from dataconfig import *
from cretio_dataloader import *
from utils import *
from dnn_model import *
import numpy as np

# keras的 标准的训练 格式 是 (data, label)， 如需特别的改，需要重写 train ， evaluate 函数
def create_tf_dataset(dataset):
    def generator():
        for batch in dataset:
            data = batch
            label = batch.pop('label')
            label = tf.expand_dims(label, 1)
            yield data, label  # 返回 data 和 label 作为一个 tuple
    return tf.data.Dataset.from_generator(generator, output_signature=(
        {
            name:tf.TensorSpec(shape=(None, ), dtype=v.dtype) for name, v in inputs.items()
        },
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    ))

np.set_printoptions(precision=4, suppress=True)



# 读取特征
spase_feature_names = DATA_CONFIG['SPARSE_FEATURES']
dense_feature_names = DATA_CONFIG['DENSE_FEATURES']
label_feature_names = DATA_CONFIG['label']

#读 tfrecords 数据
data_path = "../../data/tf_data"
dataset, valid_data, eval_data = get_cretio_data(spase_feature_names, dense_feature_names, label_feature_names, data_path)
inputs = get_cretio_input(dense_feature_names, spase_feature_names)

# 计算dense features 均值和方差
dense_feature_statics = get_dataset_statics(dataset, dense_feature_names)
# z-score 初始化
dense_layer = Dense_Process_Zscore_Layer(spase_feature_names, dense_feature_names, dense_feature_statics)
# log 初始化
# dense_layer = Dense_Process_LOG_Layer(spase_feature_names, dense_feature_names, dense_feature_statics)
# model = Cretio_DEEPFM_DNN([256,256,256], dense_layer, spase_feature_names)
model = Cretio_DEEPFM_DNN_DROPOUT([400,400,400], dense_layer, spase_feature_names, dropout_rate = 0)





# ✅ 设置 TensorBoard
# ========================
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=3,
    restore_best_weights=True
)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits= False), optimizer = 'adam',metrics=[tf.keras.metrics.AUC(name='auc')])
model.fit(create_tf_dataset(dataset), epochs = 1,  validation_data=create_tf_dataset(valid_data), callbacks = [early_stopping, tensorboard_callback])


results = model.evaluate(create_tf_dataset(eval_data))
print("Test Loss:", results[0])
print("Test AUC:", results[1])  # 如果 metrics=[AUC]，那就是第1个指标
