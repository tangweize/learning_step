import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import sys
import os
import datetime
import json



def send_filelist(msg_content, file_paths):
    subject = '实验结果' + (datetime.datetime.now().strftime('%Y-%m-%d:%H:%M'))
    # 发信方的信息：发信邮箱，QQ 邮箱授权码
    from_addr = 'tangweize'
    password = '********1******'
    # 收信方邮箱
    recievers = ['756648174@qq.com']
    # 发信服务器
    smtp_server = 'smtp.exmail.qq.com'

    # 创建邮件
    msg = MIMEMultipart()
    msg['From'] = Header('读数完成')  # 发送者
    msg['Subject'] = Header(subject, 'utf-8')  # 邮件主题
    msg.attach(MIMEText(msg_content, 'plain', 'utf-8'))

    # **遍历所有文件并添加到邮件附件**
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment', filename=file_path.split('/')[-1])
                msg.attach(part)
        except Exception as e:
            print(f"无法附加文件 {file_path}: {e}")

    # 发送邮件
    try:
        smtpobj = smtplib.SMTP_SSL(smtp_server)
        smtpobj.connect(smtp_server, 465)
        smtpobj.login(from_addr, password)
        smtpobj.sendmail(from_addr, recievers, msg.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(f"无法发送邮件: {e}")
    finally:
        smtpobj.quit()

def get_delta_date_str(date_str, delta):
    return  (datetime.datetime.strptime(date_str, "%Y-%m-%d") + datetime.timedelta(days=delta)).strftime('%Y-%m-%d')

def str2date(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")

def date2str(date):
    return date.strftime('%Y-%m-%d')

def get_train_test_data(data, startdate, enddate, testdate):
    train_data = data[
        (data['install_date'] >= startdate) & (data['install_date'] <= enddate)].reset_index(
        drop=True).copy() 

    test_data = data[
        (data['install_date'] == testdate)].reset_index(drop=True).copy()  

    return train_data, test_data

def calculate_bias(data, suffix = 'base'):
    
    channel_site_bias = data.groupby(['install_date', 'channel', 'site_name']).agg({'tweedie_predict_delta_7d':'sum', 'b2_sale_amt_7d':'sum'}).reset_index()
    dim_os_creative_bias = data.groupby(['install_date', 'channel',  'dim_os_name', 'creative_classify']).agg({'tweedie_predict_delta_7d':'sum', 'b2_sale_amt_7d':'sum'}).reset_index()
    channel_site_bias['tweedie_predict_delta_7d' + suffix] =  channel_site_bias['tweedie_predict_delta_7d']
    dim_os_creative_bias['tweedie_predict_delta_7d' + suffix] = dim_os_creative_bias['tweedie_predict_delta_7d']
    
    return channel_site_bias, dim_os_creative_bias 


def merge_results(base_res, iter_res):
    online_channel_site_bias, online_dim_os_creative_bias = calculate_bias(base_res, '_base')
    optim_channel_site_bias, optim_dim_os_creative_bias = calculate_bias(iter_res, '_iter')
    
    channel_site_bias_compare = online_channel_site_bias.merge(optim_channel_site_bias[['install_date', 'channel', 'site_name','tweedie_predict_delta_7d_iter']], 
                                                           on = ['install_date', 'channel', 'site_name', ], 
                                                           how = 'left')
    dim_os_creative_bias = online_dim_os_creative_bias.merge(optim_dim_os_creative_bias[['install_date',  'channel', 'dim_os_name', 'creative_classify', 'tweedie_predict_delta_7d_iter']], 
                                                             on = ['install_date', 'channel', 'dim_os_name', 'creative_classify' ],
                                                             how = 'left')


    channel_site_bias_compare['iter_bias'] = channel_site_bias_compare.apply(lambda row: (row['tweedie_predict_delta_7d_iter'] - row['b2_sale_amt_7d']) / ( row['b2_sale_amt_7d'] + 1), axis = 1 )
    channel_site_bias_compare['base_bias'] = channel_site_bias_compare.apply(lambda row: (row['tweedie_predict_delta_7d_base'] - row['b2_sale_amt_7d']) / ( row['b2_sale_amt_7d'] + 1), axis = 1 )

    dim_os_creative_bias['iter_bias'] = dim_os_creative_bias.apply(lambda row: round((row['tweedie_predict_delta_7d_iter'] - row['b2_sale_amt_7d']) / ( row['b2_sale_amt_7d'] + 1), 3), axis = 1 )
    dim_os_creative_bias['base_bias'] = dim_os_creative_bias.apply(lambda row: round( (row['tweedie_predict_delta_7d_base'] - row['b2_sale_amt_7d']) / ( row['b2_sale_amt_7d'] + 1), 3), axis = 1 )
    
    return channel_site_bias_compare, dim_os_creative_bias

def get_channel_bias(res, thd = -1, base_col = 'tweedie_predict_delta_7d_base', iter_col = 'tweedie_predict_delta_7d_iter', channel = ['今日头条2.0']):
    res = res[res.channel.isin(channel)]
    
    if thd >= 0:
        res = res[res.b2_sale_amt_7d >= thd]
    res = res[(res.base_bias.isna() == False) & (res.iter_bias.isna() == False)]

    res['iter_bias'] = res['iter_bias'].apply(lambda x: abs(x))
    res['base_bias'] = res['base_bias'].apply(lambda x: abs(x))
    
    
    temp = res.groupby(['install_date','channel']).agg({'b2_sale_amt_7d':'sum'})
    res = res.merge(temp.rename(columns={'b2_sale_amt_7d': 'b2_sale_amt_7d_sum'}), 
    on=['install_date','channel'], 
    how='left'
    )
    
    res['sale_weight'] = res['b2_sale_amt_7d'] / res['b2_sale_amt_7d_sum'] 
    
    by_dt_bias = res.groupby('install_date').agg({'base_bias': 'mean', 'iter_bias':'mean'})
    # print("="* 10, "平均bias","="* 10)
    # display(by_dt_bias)
    # display(by_dt_bias.agg({'base_bias': 'mean', 'iter_bias':'mean'}))
    
    
    res['sale_weight'] = res['b2_sale_amt_7d'] / res['b2_sale_amt_7d_sum'] 
    res['base_bias'] = res['base_bias'].apply(lambda x: abs(x)) *  res['sale_weight']
    res['iter_bias'] = res['iter_bias'].apply(lambda x: abs(x)) *  res['sale_weight']


    by_dt_bias = res.groupby('install_date').agg({'base_bias': 'sum', 'iter_bias':'sum'})
    
    by_dt_bias['提升'] = (by_dt_bias['iter_bias'] - by_dt_bias['base_bias']) / (by_dt_bias['base_bias'] + 0.0001)
    
    value_row = pd.DataFrame({
    'base_bias': [by_dt_bias['base_bias'].mean()],
    'iter_bias': [by_dt_bias['iter_bias'].mean()],
    '提升': [(by_dt_bias['iter_bias'].mean() - by_dt_bias['base_bias'].mean()) /( by_dt_bias['base_bias'].mean() + 0.0001) ],
    }, index=['MEAN'])
    
    by_dt_bias = pd.concat([by_dt_bias, value_row])
    
    print("="* 10, "加权bias","="* 10)
    display(by_dt_bias)
    display(by_dt_bias.agg({'base_bias': 'mean', 'iter_bias':'mean'}))
    return by_dt_bias


import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import DataBarRule

def save_excel_with_progress_bar(df, column, filename="highlighted.xlsx"):
    """
    在 Excel 里高亮指定 `column`，并为 `base_bias` 和 `iter_bias` 添加进度条，并将其格式化为百分比。
    对“提升”列的正数标柔和红色，负数标柔和绿色。

    参数:
    - df: pandas DataFrame
    - column: 需要高亮的列
    - filename: 输出 Excel 文件名 (默认: "highlighted.xlsx")
    """
    wb = Workbook()
    ws = wb.active

    # 颜色映射
    colors = ["FFDDC1", "FFABAB", "FFC3A0", "D5AAFF", "85E3FF", "B9FBC0"]
    unique_values = df[column].unique()
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}

    # 写入表头
    for col_num, column_title in enumerate(df.columns, start=1):
        ws.cell(row=1, column=col_num, value=column_title)

    # 写入数据并高亮
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        value = getattr(row, column)
        fill = PatternFill(start_color=color_map[value], end_color=color_map[value], fill_type="solid")

        for col_idx, cell_value in enumerate(row, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=cell_value)
            cell.fill = fill

            # 格式化 base_bias 和 iter_bias 为百分比
            if df.columns[col_idx-1] in ["base_bias", "iter_bias","提升"]:
                cell.number_format = '0.00%'  # 将数字格式设置为百分比，保留两位小数

            # 对 “提升” 列进行颜色标记
            if df.columns[col_idx-1] == "提升":
                if cell_value > 0:
                    cell.fill = PatternFill(start_color="FFCCCB", end_color="FFCCCB", fill_type="solid")  # 正数标柔和红色
                elif cell_value < 0:
                    cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # 负数标柔和绿色

    # 进度条：base_bias 和 iter_bias
    base_bias_col = df.columns.get_loc("base_bias") + 1
    iter_bias_col = df.columns.get_loc("iter_bias") + 1

    max_base = 1.0
    max_iter = 1.0

    base_bias_rule = DataBarRule(start_type="num", start_value=0, 
                                 end_type="num", end_value=max_base, color="63C384")
    iter_bias_rule = DataBarRule(start_type="num", start_value=0, 
                                 end_type="num", end_value=max_iter, color="FF6961")

    ws.conditional_formatting.add(f"{chr(64+base_bias_col)}2:{chr(64+base_bias_col)}{len(df)+1}", base_bias_rule)
    ws.conditional_formatting.add(f"{chr(64+iter_bias_col)}2:{chr(64+iter_bias_col)}{len(df)+1}", iter_bias_rule)

    # 保存 Excel
    wb.save(filename)
    print(f"Excel 文件已保存: {filename}")

# 示例调用
# save_excel_with_progress_bar(df, 'channel', 'highlighted_with_progress.xlsx')


def get_channel_site_bias(res, thd = -1, base_col = 'tweedie_predict_delta_7d_base', iter_col = 'tweedie_predict_delta_7d_iter', channel = ['今日头条2.0']):
    res = res[res.channel.isin(channel)]
    
    if thd >= 0:
        res = res[res.b2_sale_amt_7d >= thd]
    res = res[(res.base_bias.isna() == False) & (res.iter_bias.isna() == False)]

    res['iter_bias'] = res['iter_bias'].apply(lambda x: abs(x))
    res['base_bias'] = res['base_bias'].apply(lambda x: abs(x))
    
    temp = res.groupby(['site_name']).agg({'b2_sale_amt_7d':'sum'})
    res = res.merge(temp.rename(columns={'b2_sale_amt_7d': 'b2_sale_amt_7d_sum'}), 
    on=['site_name'], 
    how='left'
    )

    
    res['提升'] = (res['iter_bias'] - res['base_bias']) / (res['base_bias'] + 0.0001)

    res = res[res.b2_sale_amt_7d_sum > 10000]
    res = res.sort_values(by = ['b2_sale_amt_7d_sum', 'site_name', 'install_date'], ascending = False)
    by_dt_res = res
    summary_res = res.groupby(['channel', 'site_name','b2_sale_amt_7d_sum']).agg({'base_bias': 'mean', 'iter_bias':'mean'}).sort_values(by = ['b2_sale_amt_7d_sum',], ascending = False)

    return by_dt_res, summary_res.reset_index()



def get_channel_biz_os_bias(res, thd = -1, base_col = 'tweedie_predict_delta_7d_base', iter_col = 'tweedie_predict_delta_7d_iter', channel = ['今日头条2.0']):
    res = res[res.channel.isin(channel)]
    
    if thd >= 0:
        res = res[res.b2_sale_amt_7d >= thd]
    res = res[(res.base_bias.isna() == False) & (res.iter_bias.isna() == False)]

    res['iter_bias'] = res['iter_bias'].apply(lambda x: abs(x))
    res['base_bias'] = res['base_bias'].apply(lambda x: abs(x))
    
    temp = res.groupby(['dim_os_name','creative_classify']).agg({'b2_sale_amt_7d':'sum'})
    res = res.merge(temp.rename(columns={'b2_sale_amt_7d': 'b2_sale_amt_7d_sum'}), 
    on=['dim_os_name','creative_classify'], 
    how='left'
    )

    
    res['提升'] = (res['iter_bias'] - res['base_bias']) / (res['base_bias'] + 0.0001)

    res = res[res.b2_sale_amt_7d_sum > 10000]
    res = res.sort_values(by = ['b2_sale_amt_7d_sum', 'creative_classify', 'dim_os_name', 'install_date'], ascending = False)
    by_dt_res = res
    summary_res = res.groupby(['b2_sale_amt_7d_sum','dim_os_name','creative_classify']).agg({'base_bias': 'mean', 'iter_bias':'mean'}).sort_values(by = ['b2_sale_amt_7d_sum',], ascending = False)

    return by_dt_res, summary_res.reset_index()

    
def get_channel_site_bias(res, thd = -1, base_col = 'tweedie_predict_delta_7d_base', iter_col = 'tweedie_predict_delta_7d_iter', channel = ['今日头条2.0']):
    res = res[res.channel.isin(channel)]
    
    if thd >= 0:
        res = res[res.b2_sale_amt_7d >= thd]
    res = res[(res.base_bias.isna() == False) & (res.iter_bias.isna() == False)]

    res['iter_bias'] = res['iter_bias'].apply(lambda x: abs(x))
    res['base_bias'] = res['base_bias'].apply(lambda x: abs(x))
    
    temp = res.groupby(['site_name']).agg({'b2_sale_amt_7d':'sum'})
    res = res.merge(temp.rename(columns={'b2_sale_amt_7d': 'b2_sale_amt_7d_sum'}), 
    on=['site_name'], 
    how='left'
    )

    
    res['提升'] = (res['iter_bias'] - res['base_bias']) / (res['base_bias'] + 0.0001)

    res = res[res.b2_sale_amt_7d_sum > 10000]
    res = res.sort_values(by = ['b2_sale_amt_7d_sum', 'site_name', 'install_date'], ascending = False)
    by_dt_res = res
    summary_res = res.groupby(['channel', 'site_name','b2_sale_amt_7d_sum']).agg({'base_bias': 'mean', 'iter_bias':'mean'}).sort_values(by = ['b2_sale_amt_7d_sum',], ascending = False)

    return by_dt_res, summary_res.reset_index()

import pickle

def save_obj(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


import pickle

def load_obj(file_path):
    """
    使用 pickle 加载保存的 Python 对象（例如 .pkl 或 .obj 文件）

    参数:
        file_path (str): 文件路径，通常是 .pkl 或 .obj 后缀

    返回:
        object: 从文件中反序列化得到的 Python 对象
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj




def read_feature_json_config(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # 使用json.load()函数读取文件内容并将其转换为字典
        config = json.load(file)
    group_2_features = {}
    for i in config:
        group_2_features[i] = []
        for feature in config[i]:
            group_2_features[i].append(feature['name'])
    return group_2_features


import tensorflow as tf
def tfrecords_to_dataset(path, parse_function):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_function)

    dataset = dataset.prefetch(buffer_size=10000)
    dataset = dataset.batch(512)

    return dataset



def get_trian_valid_test_dateset(parse_function, train_path, valid_path = None, test_path = None ):
    train_dataset = tfrecords_to_dataset(train_path, parse_function)
    valid_dataset = None
    test_dataset = None
    if valid_path:
        valid_dataset = tfrecords_to_dataset(valid_path, parse_function)
    if test_path:
        test_dataset = tfrecords_to_dataset(test_path, parse_function)

    return train_dataset, valid_dataset, test_dataset



# group_2_features = read_feature_json_config('feature_config/feature_list.json')
