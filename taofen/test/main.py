import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import sys
sys.path.append("../utils")
sys.path.append("../data/")
from dataconfig import *
from utils import *

# feature 分类
spase_feature_names = DATA_CONFIG['SPARSE_FEATURES']
dense_feature_names = DATA_CONFIG['DENSE_FEATURES']
label_feature_names = DATA_CONFIG['label']

dataset = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord("../data/tf_data/train_criteo_200w_rows.tfrecord", 512)

# 定义 inputs
inputs = { name:keras.Input(shape=(1,), name=name, dtype=tf.float32) for name in dense_feature_names }
inputs.update({
    name:keras.Input(shape=(1,), name=name, dtype=tf.string) for name in spase_feature_names
})


numeric_inputs = { name:value for name, value in inputs.items() if value.dtype == tf.float32 }
print("numeric",numeric_inputs.values())
x = layers.Concatenate()(list(numeric_inputs.values()))
# 用log 处理？
# norm = layers.Normalization()
# norm.adapt(np.array(titanic[numeric_inputs.keys()]))
# x = norm(x)
preprocessed_inputs = [x]
num_bins = 10000
embedding_dim = 8
sparse_inputs = { name:value for name, value in inputs.items() if value.dtype == tf.string }
for name, input in sparse_inputs.items():
    #是否能跑通？
    lookup = keras.layers.Hashing(num_bins=num_bins)
    embedding_layer = layers.Embedding(input_dim=num_bins, output_dim=8)
    x = lookup(input)
    x = embedding_layer(x)
    x = tf.squeeze(x, axis = 1)
    preprocessed_inputs.append(x)


results = layers.Concatenate()(preprocessed_inputs)
process_model = keras.Model(inputs, results)
tf.keras.utils.plot_model(model = process_model, rankdir='LR',dpi = 75,
                         show_shapes=True )


