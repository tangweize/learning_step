
import pandas as pd
import numpy as np
from data_loader import *
from feature_engineering import  *

def process_label(df, label_col, diff_col, target):
    df[target] = df[label_col] - df[diff_col]
    return df




# 参数
FILEPATH = 'data/ad_install_ltv_sequence_feature.obj'
DATE_CUTOFF = '2025-05-10'
promo_dates = [

    {"date": "2022-02-01", "festival": "春节"},
    {"date": "2022-02-14", "festival": "情人节"},
    {"date": "2022-03-08", "festival": "女王节"},
    {"date": "2022-05-20", "festival": "520表白节"},
    {"date": "2022-06-18", "festival": "618年中大促"},
    {"date": "2022-08-04", "festival": "七夕节"},
    {"date": "2022-09-10", "festival": "中秋节"},
    {"date": "2022-10-01", "festival": "国庆大促"},
    {"date": "2022-11-11", "festival": "双11"},
    {"date": "2022-12-12", "festival": "双12"},
    {"date": "2022-12-25", "festival": "圣诞节"},

    {"date": "2023-01-22", "festival": "春节"},
    {"date": "2023-02-14", "festival": "情人节"},
    {"date": "2023-03-08", "festival": "女王节"},

    {"date": "2023-05-20", "festival": "520表白节"},
    {"date": "2023-06-18", "festival": "618年中大促"},
    {"date": "2023-08-22", "festival": "七夕节"},
    {"date": "2023-09-29", "festival": "中秋节"},
    {"date": "2023-10-01", "festival": "国庆大促"},
    {"date": "2023-11-11", "festival": "双11"},
    {"date": "2023-12-12", "festival": "双12"},
    {"date": "2023-12-25", "festival": "圣诞节"},

    {"date": "2024-02-10", "festival": "春节"},
    {"date": "2024-02-14", "festival": "情人节"},
    {"date": "2024-03-08", "festival": "女王节"},

    {"date": "2024-05-20", "festival": "520表白节"},
    {"date": "2024-06-18", "festival": "618年中大促"},
    {"date": "2024-08-10", "festival": "七夕节"},
    {"date": "2024-09-17", "festival": "中秋节"},
    {"date": "2024-10-01", "festival": "国庆大促"},
    {"date": "2024-11-11", "festival": "双11"},
    {"date": "2024-12-12", "festival": "双12"},
    {"date": "2024-12-25", "festival": "圣诞节"},

    {"date": "2025-01-29", "festival": "春节"},
    {"date": "2025-02-14", "festival": "情人节"},
    {"date": "2025-03-08", "festival": "女王节"},
    {"date": "2025-05-20", "festival": "520表白节"},
    {"date": "2025-06-18", "festival": "618年中大促"},
    {"date": "2025-07-29", "festival": "七夕节"},
    {"date": "2025-10-06", "festival": "中秋节"},
    {"date": "2025-10-01", "festival": "国庆大促"},
    {"date": "2025-11-11", "festival": "双11"},
    {"date": "2025-12-12", "festival": "双12"},
    {"date": "2025-12-25", "festival": "圣诞节"},
]

PROMO_DATES = [ pdate['date'] for pdate in promo_dates if pdate['festival'] in  ('618年中大促', '双11','双12', '圣诞节')]

HOLIDAY_DATES = [
    '2022-01-01','2022-02-01',  '2022-05-01', '2022-06-03', '2022-09-10', '2022-10-01', '2022-10-04', '2022-01-31',
    '2023-01-01', '2023-01-22',  '2023-05-01', '2023-06-22', '2023-09-29', '2023-10-01', '2023-10-23', '2023-01-21',
    '2024-01-01','2024-02-10',  '2024-05-01', '2024-06-10', '2024-09-17', '2024-10-01', '2024-10-02', '2024-02-09',
    '2025-01-01','2025-01-29',  '2025-05-01', '2025-06-20', '2025-10-06', '2025-10-01', '2025-09-30', '2025-01-28'
]

# 加载数据
df = load_data(FILEPATH, DATE_CUTOFF)
df['diff_0'] = 0
df = df[df.dim_ad_channel_level1 == '信息流']


df = add_date_features(df, PROMO_DATES, HOLIDAY_DATES)
# 特征处理
categorical_features = ['month','weekday','is_holiday','is_promo','season']
dense_features = ['install_deviceid_cnt','day1_ltv_sum','days_to_next_promo','days_past_last_promo']


# label处理
target = 'day7_ltv_label'
row_label = 'day7_ltv_sum'
diff_label = 'diff_0'   # 'diff_0', 'day1_ltv_sum'
df = process_label(df, row_label, diff_label, target)

print(df)
#
# with torch.no_grad():
#     y_pred = model(X_test_cat, X_test_dense).squeeze().numpy()
#
#
# mse, mae, y_true_orig, y_pred_orig = evaluate_predictions(y_test, y_pred, scaler_y, y_test_diff_label)
# print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")
#
# relative_error = (y_pred_orig - y_true_orig ) / y_true_orig
#
#
# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# start_date = '2024-01-15'
# end_date = '2025-05-04'
# show_all_xticks = False
#
# highlight_dates = ['2024-02-01','2024-02-18','2025-01-21','2025-02-05']
#
#
# plot_predictions(test_dates, y_true_orig, y_pred_orig, start_date, end_date, show_all_xticks, ax=axes[0],highlight_dates=highlight_dates)
# plot_relative_error(test_dates, relative_error, start_date, end_date, show_all_xticks, ax=axes[1],highlight_dates=highlight_dates)
#
# plt.tight_layout()
# plt.show()
#

# import matplotlib.pyplot as plt
#
# fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# start_date = '2024-02-01'
# end_date = '2024-02-15'
# show_all_xticks = True
#
# plot_predictions(test_dates, y_true_orig, y_pred_orig, start_date, end_date, show_all_xticks, ax=axes[0])
# plot_relative_error(test_dates, relative_error, start_date, end_date, show_all_xticks, ax=axes[1])
#
# plt.tight_layout()
# plt.show()
