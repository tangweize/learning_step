import sys
import os
import json
import tensorflow as tf
import tqdm
from models.model import *
import numpy as np
from sklearn.isotonic import IsotonicRegression
import tf2onnx
import onnxruntime as ort
import shutil
from ltv_utils import *
from losses.custom_loss import *
pd.set_option('display.float_format', '{:.4f}'.format)  # 保留10位小数，可调整
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def parse_function(serialized_example):
    feature_description = {
        'deviceid': tf.io.FixedLenFeature([], tf.string),
        'install_date': tf.io.FixedLenFeature([], tf.string),
        'dim_os_name1': tf.io.FixedLenFeature([], tf.string),
        'creative_classify1': tf.io.FixedLenFeature([], tf.string),
        'total_pay_amount1':  tf.io.FixedLenFeature([], tf.float32),
         'channel1': tf.io.FixedLenFeature([], tf.string),
        'b2_sale_amt_bias':  tf.io.FixedLenFeature([], tf.int64),
         'b2_sale_amt_7d': tf.io.FixedLenFeature([], tf.int64),
         'install_time': tf.io.FixedLenFeature([], tf.string),
        'install_order_diff':  tf.io.FixedLenFeature([], tf.int64),
        'all_install_order_7d_diff':  tf.io.FixedLenFeature([], tf.int64),
        'is_a1x_a33':  tf.io.FixedLenFeature([], tf.int64),
        'platform_label':  tf.io.FixedLenFeature([], tf.string),
        'user_dense_price_features': tf.io.FixedLenFeature([len(group_2_features['user_dense_price_features'])], tf.float32),
        'user_dense_duration_features': tf.io.FixedLenFeature([len(group_2_features['user_dense_duration_features'])], tf.float32),
        'user_dense_features': tf.io.FixedLenFeature([len(group_2_features['user_dense_features'])], tf.float32),
        'user_sparse_features': tf.io.FixedLenFeature([len(group_2_features['user_sparse_features'])], tf.float32)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example

def create_tf_dataset(dataset):
    sample_batch = next(iter(dataset))
    sample_data = {k: v for k, v in sample_batch.items() if k not in ['b2_sale_amt_7d', 'total_pay_amount1']}

    def generator():
        for batch in dataset:
            hour = tf.cast(tf.gather(batch['user_sparse_features'], indices=0, axis=1) - 1,
                           tf.int64)  # shape: (batch_size,)
            b2_7d = tf.cast(tf.reshape(batch.pop('b2_sale_amt_7d'), (-1, 1)), tf.float32)
            # 将 b2_7d 中小于 0 的值替换为 0
            b2_7d = tf.maximum(b2_7d, 0.0)

            total_amt_1h = tf.reshape(batch.pop('total_pay_amount1'), (-1, 1))

            # 将保留的样本和标签一起返回
            y_true_packed = tf.concat([b2_7d, total_amt_1h], axis=1)

            # y_true_packed = b2_7d
            yield batch, y_true_packed

    # 正确写法：output_signature 中保留每个字段的真实 shape
    output_signature = (
        {
            name: tf.TensorSpec(shape=(None,) + v.shape[1:], dtype=v.dtype)
            for name, v in sample_data.items()
        },
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

def save_onnx(model_add_ir, tmp_path, target_path, model_file_name):

    tmp_model_path = tmp_path + model_file_name
    onnx_model, _ = tf2onnx.convert.from_keras(model_add_ir, input_signature=spec, opset=13)

    with open(tmp_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        # 创建保存路径（如果不存在）

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    model_path = target_path + model_name
    # 发布到线上路径
    shutil.move(tmp_model_path, model_path)

    # 存档路径
    shutil.move(tmp_model_path, datetime.datetime.now().strftime('%Y%m%d') + "multi_window_liuliang_ltv_predict_model_v1" +  model_path)


if __name__ == "__main__":

    # step1： 训练 多头深度模型
    group_2_features = read_feature_json_config('features/feature_list.json')
    train_file_name = '/tf/hdfs/data/big_data_multi_window_model_train/part-r-00000'
    valid_file_name = '/tf/hdfs/data/big_data_multi_window_model_valid/part-r-00000'
    test_file_name = '/tf/hdfs/data/big_data_multi_window_model_test/part-r-00000'

    train_dataset, valid_dataset, test_dataset = get_trian_valid_test_dateset(parse_function, 10240, train_file_name, valid_file_name, test_file_name)



    user_dense_price_features = group_2_features['user_dense_price_features']
    user_dense_duration_features = group_2_features['user_dense_duration_features']
    user_dense_features = group_2_features['user_dense_features']
    user_sparse_features = group_2_features['user_sparse_features']



    emb_features = [
    'creative_classify','dim_device_manufacture', 'car_add_type_most','show_order_is_2arrival_latest', 'selecttirecount_most', 'show_order_is_2arrival_most','selecttirecount_latest',
     'new_sitename','advsite','car_add_type_latest','platform_level', 'tire_list_click_avg_index','tire_list_click_most_pid_level','tire_order_page_most_pid_level',
    ]

    dnn_param = {"use_bn": False, "drop_out":0.2}

    model = MULTI_HEAD_LTV_MODEL_ADD_HEAD_BN(5, [200, 200],[128,128], 'user_dense_features', 'user_dense_price_features', 'user_dense_duration_features',
                                'user_sparse_features',user_sparse_features, emb_features, dnn_param = dnn_param)


    sample = next(iter(train_dataset))
    input_shape = {k: v.shape for k, v in sample.items()}


    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    loss_fn = UnifiedLTVLoss('delta')
    model.compile(loss=loss_fn,
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
                 )

    # train model
    model.fit(
        create_tf_dataset(train_dataset),
        epochs=20,
        validation_data = create_tf_dataset(valid_dataset),
        callbacks= [early_stopping]
    )

    exp_pred_res = model.evaluate_exp(create_tf_dataset(valid_dataset))
    rank_pred_res = model.evaluate_rank(create_tf_dataset(valid_dataset), is_plot = False)
    log_print(f"深度模型对同分布的valid集的bias : {exp_pred_res['bias']}")
    log_print(f"深度模型对同分布的valid集的auc : {rank_pred_res}")

    log_print(f"{'='*40}")


    exp_pred_res = model.evaluate_exp(create_tf_dataset(test_dataset))
    rank_pred_res = model.evaluate_rank(create_tf_dataset(test_dataset), is_plot = False)
    log_print(f"深度模型对同分布的test集的bias : {exp_pred_res['bias']}")
    log_print(f"深度模型对同分布的test集的auc : {rank_pred_res}")


    train_file_name = '/tf/hdfs/data/is_multi_window_model_data/part-r-00000'
    test_file_name = '/tf/hdfs/data/recent_thd_cal_dataset/part-r-00000'

    recent_train_dataset, _, recent_test_dataset = get_trian_valid_test_dateset(parse_function, 10000, train_file_name,
                                                                                test_file_name, test_file_name)


    # step2 保序回归

    predict_res, true_res = model.predict_head_score(create_tf_dataset(recent_train_dataset))
    head_isoreg = {}
    for head_num in predict_res.keys():
        pred, true = predict_res[head_num], true_res[head_num]
        pred_np = tf.squeeze(pred).numpy()
        true_np = tf.squeeze(true).numpy()
        iso_model = IsotonicRegression(out_of_bounds='clip')  # 防止预测越界
        iso_model.fit(pred_np, true_np)
        head_isoreg[head_num] = iso_model


    test_predict_res, test_true_res = model.predict_head_score(create_tf_dataset(recent_test_dataset))

    for head_num in  test_predict_res.keys():
        pred, true = test_predict_res[head_num], test_true_res[head_num]
        pred_np = tf.squeeze(pred).numpy()
        true_np = tf.squeeze(true).numpy()

        iso_model = head_isoreg[head_num]
        iso_pred = iso_model.predict(pred_np)
        # pred
        log_print(f"head: {head_num}, pred: {iso_pred.sum()}, true: {true_np.sum() }, bias: ",round((iso_pred.sum() - true_np.sum()) / true_np.sum(), 3))

    # step3 融入到 多头深度模型 + 保序回归
    model_add_ir = MultiHeadCalibratedLTVModel(5, [200, 200], [128, 128], 'user_dense_features',
                                               'user_dense_price_features', 'user_dense_duration_features',
                                               'user_sparse_features', user_sparse_features, emb_features,
                                               dnn_param=dnn_param)

    model_add_ir.set_calibrators(head_isoreg, user_dense_price_features, 'user_dense_price_features',
                                 'total_pay_amount')
    raw_dnn_weights = model.get_weights()

    loss_fn = UnifiedLTVLoss('delta')
    model_add_ir.compile(loss=loss_fn,
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
                         )
    model_add_ir.fit(
        create_tf_dataset(train_dataset.take(1)),
        epochs=1,
    )
    model_add_ir.set_weights(raw_dnn_weights)

    # step4 将模型 存储成 onnx
    spec = ({
                'user_dense_duration_features': tf.TensorSpec(shape=(None, 6), dtype=tf.float32,
                                                              name="user_dense_duration_features"),
                'user_dense_features': tf.TensorSpec(shape=(None, 83), dtype=tf.float32, name="user_dense_features"),
                'user_dense_price_features': tf.TensorSpec(shape=(None, 9), dtype=tf.float32,
                                                           name="user_dense_price_features"),
                'user_sparse_features': tf.TensorSpec(shape=(None, 22), dtype=tf.float32, name="user_sparse_features")
            },)




    tmp_model_path_suffix = '/tf/'
    model_path_suffix = '/root/hdfs/write/models/model/'
    model_name = 'model.onnx'

    save_onnx(model_add_ir,tmp_model_path_suffix, model_path_suffix, model_name)

    # step5: recent 样本打分存到表中
    tmp_model_path = tmp_model_path_suffix + model_name
    ort_session = ort.InferenceSession(tmp_model_path)
        # 获得模型输入特征名
    input_names = [inp.name for inp in ort_session.get_inputs()]
    log_print("模型输入名称:", input_names)
    output_names = [out.name for out in ort_session.get_outputs()]


    all_dfs = []  # 存放每个 batch 的 DataFrame
    for batch, y in create_tf_dataset(recent_test_dataset):
        inputs = {f'{k}': v.numpy() for k, v in batch.items() if k in spec[0].keys()}

        # 提取字段并转成 string（注意 bytes 转 string）
        deviceids = batch['deviceid'].numpy().astype(str)
        install_dates = batch['install_date'].numpy().astype(str)
        platform_labels = batch['platform_label'].numpy().astype(str)
        dim_os_names = batch['dim_os_name1'].numpy().astype(str)
        is_a1x_a33 = batch['is_a1x_a33'].numpy().astype(str)

        # ONNX 推理
        predict_score = ort_session.run(output_names, inputs)

        df = pd.DataFrame({
            'deviceid': deviceids,
            'install_date': install_dates,
            'platform_label': platform_labels,
            'dim_os_name1': dim_os_names,
            'is_a1x_a33': is_a1x_a33,
            'predict_score': predict_score[0].reshape(-1)
        })

        all_dfs.append(df)

    # 合并所有 batch 的 DataFrame
    final_df = pd.concat(all_dfs, ignore_index=True)
