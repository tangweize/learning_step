import tensorflow as tf
import pandas as pd
from tqdm import tqdm
class DataUtil:
    # 初始化 数据的一些配置，文件路径， 类别型特征，数值型特征和label 名
    def __init__(self, sparse_feature_names, dense_feature_names, label_name, chunk_size = 100000):
        self.chunk_size = chunk_size
        self.sparse_cols = sparse_feature_names
        self.dense_cols = dense_feature_names
        self.label_cols = label_name if isinstance(label_name, list ) else [label_name]

    def make_example(self, line):
        features = {feat: tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][feat]])) for feat in
                    self.dense_cols}
        features.update(
            {feat: tf.train.Feature(bytes_list=tf.train.BytesList(value=[line[1][feat].encode('utf-8')])) for feat in self.sparse_cols})

        features[self.label_cols[0]] = tf.train.Feature(float_list=tf.train.FloatList(value=[line[1][self.label_cols[0]]]))
        return tf.train.Example(features=tf.train.Features(feature=features))


    def write_tfrecord(self, read_path, save_path, start, end, spase_fillna = 'NaN', dense_fillna = -1):
        df = pd.read_csv(read_path, chunksize = self.chunk_size, delimiter = '\t', names =  self.label_cols + self.dense_cols + self.sparse_cols )

        writer = tf.io.TFRecordWriter(save_path)
        temp_rows = 0

        for batch in df:
            # 填补缺失值，否则 没法存成tfrecords
            batch.loc[:, self.sparse_cols] = batch[self.sparse_cols].fillna(spase_fillna)
            batch.loc[:, self.dense_cols] = batch[self.dense_cols].fillna(dense_fillna)

            if temp_rows >= start and temp_rows < end:
                total_rows = len(batch)
                for line in tqdm(batch.iterrows(), total=total_rows):
                    ex = self.make_example(line)
                    writer.write(ex.SerializeToString())
            elif temp_rows > end:
                break
            temp_rows += self.chunk_size

        writer.close()

    def read_tfrecord(self):
        return -1


# write_tfrecord('./criteo_sample.tr.tfrecords',train,sparse_features,dense_features,'label')
# write_tfrecord('./criteo_sample.te.tfrecords',test,sparse_features,dense_features,'label')
