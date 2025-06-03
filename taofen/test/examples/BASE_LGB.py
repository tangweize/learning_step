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

np.set_printoptions(precision=4, suppress=True)



# 读取特征
spase_feature_names = DATA_CONFIG['SPARSE_FEATURES']
dense_feature_names = DATA_CONFIG['DENSE_FEATURES']
label_feature_names = DATA_CONFIG['label']

#读 tfrecords 数据
data_path = "../../data/tf_data"
dataset, valid_data, eval_data = get_cretio_data(spase_feature_names, dense_feature_names, label_feature_names, data_path)





