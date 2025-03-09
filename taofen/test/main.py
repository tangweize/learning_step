import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras import layers
import pandas as pd
import sys
sys.path.append("../utils")
sys.path.append("../data/")
from dataconfig import *
from utils import *



spase_feature_names = DATA_CONFIG['SPARSE_FEATURES']
dense_feature_names = DATA_CONFIG['DENSE_FEATURES']
label_feature_names = DATA_CONFIG['label']

# 定义 inputs
dataset = DataUtil(spase_feature_names, dense_feature_names, label_feature_names).read_tfrecord("../data/tf_data/train_criteo_200w_rows.tfrecord", 512)
numeric_inputs = { name:keras.Input(shape=(1,), name=name, dtype=tf.float32) for name in dense_feature_names }
x = layers.Concatenate()(list(numeric_inputs.values()))
# 用log 处理？
# norm = layers.Normalization()
# norm.adapt(np.array(titanic[numeric_inputs.keys()]))
# x = norm(x)
preprocessed_inputs = [x]
num_bins = 10000
embedding_dim = 8
sparse_inputs = {name:keras.Input(shape=(1, ), name = name, dtype=tf.string) for name in spase_feature_names}
for name, input in sparse_inputs:
    #是否能跑通？
    lookup = layers.hashing(num_bins=num_bins)
    embedding_layer = layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=)

    preprocessed_inputs.append(embedding_layer(lookup(input)))

results = layers.Concatenate()(preprocessed_inputs)
process_model = keras.Model(inputs, results)



