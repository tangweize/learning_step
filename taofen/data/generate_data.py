import pandas as pd
import sys
from dataconfig import DATA_CONFIG
import tensorflow as tf
sys.path.append("../utils")
from utils import *

spase_feature_names = DATA_CONFIG['SPARSE_FEATURES']
dense_feature_names = DATA_CONFIG['DENSE_FEATURES']
label_feature_names = DATA_CONFIG['label']
data_file = "../data/local_data/train.txt"
# dataload = DataUtil(data_file, spase_feature_names, dense_feature_names, label_feature_names)
# dataload.write_tfrecord("../data/tf_data/train_criteo_200w_rows.tfrecord", 0, 2000000)
# dataload.write_tfrecord("../data/tf_data/valid_criteo_20w_rows.tfrecord", 2500000, 2700000)
# dataload.write_tfrecord("../data/tf_data/test_criteo_20w_rows.tfrecord", 3000000, 3500000)

dataload = DataUtil(data_file, spase_feature_names, dense_feature_names, label_feature_names, 10000)
dataload.write_tfrecord("../data/tf_data/train_criteo_5w_rows.tfrecord", 0, 50000)
dataload.write_tfrecord("../data/tf_data/valid_criteo_1w_rows.tfrecord", 100000, 110000)
dataload.write_tfrecord("../data/tf_data/test_criteo_1w_rows.tfrecord", 120000, 130000)

