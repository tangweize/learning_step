# Author: tangweize
# Date: 2025/3/18 21:47
# Description: 
# Data Studio Task:
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EveryDayClassify(nn.Module):
    def __init__(self, input_size):
        super(EveryDayClassify, self).__init__()
        self.shared1 = nn.Linear(input_size, 256)
        self.shared2 = nn.Linear(256, 128)
        self.shared3 = nn.Linear(128, 64)

        self.day1 = nn.Linear(64, 1)
        self.day2 = nn.Linear(64, 1)
        self.day3 = nn.Linear(64, 1)
        self.day4 = nn.Linear(64, 1)
        self.day5 = nn.Linear(64, 1)
        self.day6 = nn.Linear(64, 1)
        self.day7 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.shared1(x))
        x = self.relu(self.shared2(x))
        x = self.relu(self.shared3(x))

        logit_7 = self.day7(x)

        logit_1 = self.day1(x) * logit_7
        logit_2 = self.day2(x) * logit_7
        logit_3 = self.day3(x) * logit_7
        logit_4 = self.day4(x) * logit_7
        logit_5 = self.day5(x) * logit_7
        logit_6 = self.day6(x) * logit_7

        logits = torch.cat([logit_1, logit_2, logit_3, logit_4, logit_5, logit_6, logit_7], dim=1)
        return logits


def create_data_loader(input_features, input_labels):
    dataset = TensorDataset(input_features, input_labels)
    dataloader = DataLoader(dataset, batch_size=5000, shuffle=True)
    return dataloader


def calculate_gap(y_hat, y, is_abs=False):
    y_hatsum = torch.sum(y_hat.cpu(), axis=0)
    y_sum = torch.sum(y.cpu(), axis=0)
    gap = (y_hatsum - y_sum) / y_sum
    if is_abs:
        return abs(gap)
    return gap


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.bceLoss = nn.BCEWithLogitsLoss()

    def forward(self, y_hat, y, mask):
        sum_loss = 0.0

        for i, v in enumerate(mask):
            if v == 1:
                tp_yhat = torch.unsqueeze(y_hat[:, i], dim=1)
                tp_y = torch.unsqueeze(y[:, i], dim=1)
                sum_loss += self.bceLoss(tp_yhat, tp_y)
        return sum_loss


def rawdata2data(train_data, features, label_col, is_return_data_loader=True):
    train_data[features] = train_data[features].apply(lambda x: (x - x.mean()) / x.std())
    train_data[features] = train_data[features].fillna(0)
    train_features = torch.tensor(train_data[features].values, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_data[label_col].values, dtype=torch.float32).to(device)
    if is_return_data_loader:
        data_loader1 = create_data_loader(train_features, train_labels)
        return data_loader1
    return train_features, train_labels
