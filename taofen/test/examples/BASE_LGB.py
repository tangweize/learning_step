import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import sys
import platform

sys.path.append("../../utils")
sys.path.append("../../data/")
sys.path.append("../../models/")

from dataconfig import *
from cretio_dataloader import *

np.set_printoptions(precision=4, suppress=True)

# Step 1. 准备配置
sparse_feature_names = DATA_CONFIG['SPARSE_FEATURES']
dense_feature_names = DATA_CONFIG['DENSE_FEATURES']
label_col = DATA_CONFIG['label'][0]
data_path = "../../data/tf_data"

# Step 2. 加载 TFRecord 数据（训练、验证、测试）
dataset, valid_data, eval_data = get_cretio_data(sparse_feature_names, dense_feature_names, label_col, data_path)

# Step 3. TFRecord 数据转换为 numpy
def tf_dataset_to_numpy(tf_dataset, dense_cols, sparse_cols, label_name):
    dense_features = []
    sparse_features = []
    labels = []

    for batch in tf_dataset:
        dense = np.stack([batch[col].numpy() for col in dense_cols], axis=1)
        sparse = np.stack([batch[col].numpy() for col in sparse_cols], axis=1).astype(str)
        label = batch[label_name].numpy()

        dense_features.append(dense)
        sparse_features.append(sparse)
        labels.append(label)

    X_dense = np.vstack(dense_features)
    X_sparse = np.vstack(sparse_features)
    y = np.concatenate(labels)
    return X_dense, X_sparse, y

# 转换训练、验证、测试集
X_train_dense, X_train_sparse, y_train = tf_dataset_to_numpy(dataset, dense_feature_names, sparse_feature_names, label_col)
X_valid_dense, X_valid_sparse, y_valid = tf_dataset_to_numpy(valid_data, dense_feature_names, sparse_feature_names, label_col)
X_test_dense,  X_test_sparse,  y_test  = tf_dataset_to_numpy(eval_data, dense_feature_names, sparse_feature_names, label_col)

# Step 4. 构造 pandas dataframe
X_train = pd.DataFrame(np.hstack([X_train_dense, X_train_sparse]), columns=dense_feature_names + sparse_feature_names)
X_valid = pd.DataFrame(np.hstack([X_valid_dense, X_valid_sparse]), columns=dense_feature_names + sparse_feature_names)
X_test  = pd.DataFrame(np.hstack([X_test_dense,  X_test_sparse]),  columns=dense_feature_names + sparse_feature_names)

# Step 5. LabelEncode 稀疏特征
for col in sparse_feature_names:
    le = LabelEncoder()
    full_col_data = X_train[col].astype(str)
    le.fit(full_col_data)

    X_train[col] = le.transform(X_train[col].astype(str))
    X_valid[col] = le.transform(X_valid[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))

# Step 6. LightGBM 数据构造
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=sparse_feature_names)
lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train, categorical_feature=sparse_feature_names)

# Step 7. 模型训练
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=50)

# Step 8. 评估 AUC
y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
valid_auc = roc_auc_score(y_valid, y_valid_pred)
print("Validation AUC:", round(valid_auc, 4))

y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
test_auc = roc_auc_score(y_test, y_test_pred)
print("Test AUC:", round(test_auc, 4))