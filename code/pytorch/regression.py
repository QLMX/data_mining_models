#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-08 13:35
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import gc
import time

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable


## load data
train_data = pd.read_csv('../../data/train.csv')
test_data = pd.read_csv('../../data/test.csv')
epochs = 20
batch_size = 1024
learning_rate = 0.002


## category feature one_hot
test_data['label'] = -1
data = pd.concat([train_data, test_data])
cate_feature = ['gender', 'cell_province', 'id_province', 'id_city', 'rate', 'term']
for item in cate_feature:
    data[item] = LabelEncoder().fit_transform(data[item])
    item_dummies = pd.get_dummies(data[item])
    item_dummies.columns = [item + str(i + 1) for i in range(item_dummies.shape[1])]
    data = pd.concat([data, item_dummies], axis=1)
data.drop(cate_feature,axis=1,inplace=True)

train = data[data['label'] != -1]
test = data[data['label'] == -1]

##Clean up the memory
del data, train_data, test_data
gc.collect()

## get train feature
del_feature = ['auditing_date', 'due_date', 'label']
features = [i for i in train.columns if i not in del_feature]


## Convert the label to two categories
train_x = train[features]
train_y = train['label'].values
test = test[features]

## Fill missing value
for i in train_x.columns:
    # print(i, train_x[i].isnull().sum(), test[i].isnull().sum())
    if train_x[i].isnull().sum() != 0:
        train_x[i] = train_x[i].fillna(-1)
        test[i] = test[i].fillna(-1)

## normalized
scaler = StandardScaler()
train_X = scaler.fit_transform(train_x)
test_X = scaler.transform(test)

class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden//2)

        self.hidden_3 = torch.nn.Linear(n_hidden//2, n_hidden//4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden//4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden//8, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))
        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))
        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))
        x = F.relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.dropout(self.bn4(x))
        x = self.out(x)
        return x

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
NN_predictions = np.zeros((test_X.shape[0], 1))
oof_preds = np.zeros((train_X.shape[0], 1))

x_test = np.array(test_X)
x_test = torch.tensor(x_test, dtype=torch.float)
if torch.cuda.is_available():
    x_test = x_test.cuda()
test = TensorDataset(x_test)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []
train_y = np.log1p(train_y)   # Data smoothing


for fold_, (trn_, val_) in enumerate(folds.split(train_x)):
    print("fold {}".format(fold_ + 1))

    x_train = Variable(torch.Tensor(train_X[trn_.astype(int)]))
    y_train = Variable(torch.Tensor(train_y[trn_.astype(int), np.newaxis]))

    x_valid = Variable(torch.Tensor(train_X[val_.astype(int)]))
    y_valid = Variable(torch.Tensor(train_y[val_.astype(int), np.newaxis]))

    model = MLP(x_train.shape[1], 512, 1, dropout=0.3)

    if torch.cuda.is_available():
        x_train, y_train = x_train.cuda(), y_train.cuda()
        x_valid, y_valid = x_valid.cuda(), y_valid.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = torch.nn.L1Loss()

    train = TensorDataset(x_train, y_train)
    valid = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()       # clear gradients for next train
            loss.backward()             # -> accumulates the gradient (by addition) for each parameter
            optimizer.step()            # -> update weights and biases
            avg_loss += loss.item() / len(train_loader)
            # avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)
        model.eval()

        valid_preds_fold = np.zeros((x_valid.size(0), 1))
        test_preds_fold = np.zeros((len(test_X), 1))

        avg_val_loss = 0.
        # avg_val_auc = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            # avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            tt = y_pred.cpu().numpy()
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)

    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i + 1) * batch_size] = y_pred.cpu().numpy()

    oof_preds[val_] = valid_preds_fold
    NN_predictions += test_preds_fold / folds.n_splits

result = NN_predictions[:, 0]
result = np.expm1(result)  #reduction
