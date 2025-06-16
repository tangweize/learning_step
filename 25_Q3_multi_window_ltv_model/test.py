import sys
import os
import json
import tensorflow as tf
import tqdm

from ltv_utils import *
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
        'dense_vector': tf.io.FixedLenFeature([dense_vector_length], tf.float32),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    return example


def read_datafrom_tf(file):
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(parse_function)
    # 将数据转换为Python字典列表
    dataset = dataset.prefetch(buffer_size=1000)

    data_list = []
    for example in tqdm.tqdm(dataset):
        data_dict = {}
        for key in example.keys():
            if key in ['dense_vector', 'sale_amt_7d_label', 'sale_amt_in_1h1', 'b2_sale_amt_1d1']:
                # 将dense_vector从Tensor转换为NumPy数组
                data_dict[key] = example[key].numpy()
            elif key in ["dt", "deviceid", "install_date", "channel1", 'dim_os_name1', 'creative_classify1',
                         'adaccountid1', 'site_name1', 'install_time']:
                data_dict[key] = example[key].numpy().decode('utf-8')  # 将bytes转换为字符串
            else:
                data_dict[key] = example[key].numpy()

        data_list.append(data_dict)

    return data_list


from multiprocessing import Pool
import psutil

dense_vector_length = 121
if __name__ == '__main__':
    train_file_dir = None

    file_name = 'ltv_0610_user_sample_to_158'

    save_path = './data/ltv_predict_sample_obj'

    train_file_dir = None
    all_train_dt = []
    data_path = f'./data/{file_name}/'
    filenames = os.listdir(data_path)
    for filename in filenames:
        if "part" in filename:
            train_file_dir = os.path.join(data_path, filename)
            all_train_dt.append(train_file_dir)

    print(all_train_dt)
    with Pool(1) as p:
        results = p.map(read_datafrom_tf, all_train_dt[:1])

    # 转成 pandas
    df = None
    for res in results:
        if df is None:
            df = pd.DataFrame(res)
        else:
            df = df.append(res)

    df = df.reset_index(drop=True)

    save_obj(df[:100000], "data/test_data.obj")