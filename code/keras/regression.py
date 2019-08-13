#!usr/bin/env python  
#-*- coding:utf-8 _*- 
"""
@version: python3.6
@author: QLMX
@contact: wenruichn@gmail.com
@time: 2019-08-01 23:10
公众号：AI成长社
知乎：https://www.zhihu.com/people/qlmx-61/columns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import gc

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter

## load data
train_data = pd.read_csv('../../data/train.csv')
test_data = pd.read_csv('../../data/test.csv')
epochs = 3
batch_size = 1024
classes = 1

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

train_x = train[features]
train_y = train['label'].values
test = test[features]

## Fill missing value
for i in train_x.columns:
    # print(i, train_x[i].isnull().sum(), test[i].isnull().sum())
    if train_x[i].isnull().sum() != 0:
        train_x[i].fillna(-1, inplace=True)
        test[i].fillna(-1, inplace=True)

## normalized
scaler = StandardScaler()
train_X = scaler.fit_transform(train_x)
test_X = scaler.transform(test)

K.clear_session()
def MLP(dropout_rate=0.25, activation='relu'):
    start_neurons = 512
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=train_X.shape[1], activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(classes, activation='linear'))
    return model


def plot_loss_acc(history, fold):
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.savefig('../../result/model_loss' + str(fold) + '.png')
    plt.show()

    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.savefig('../../result/model_accuracy' + str(fold) + '.png')
    plt.show()


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
NN_predictions = np.zeros((test_X.shape[0], 1))
oof_preds = np.zeros((train_X.shape[0], 1))

patience = 50   ## How many steps to stop
call_ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1,
                                        mode='auto', baseline=None)

for fold_, (trn_, val_) in enumerate(folds.split(train_x)):
    print("fold {}".format(fold_ + 1))
    x_train, y_train = train_X[trn_], train_y[trn_]
    x_valid, y_valid = train_X[val_], train_y[val_]


    model = MLP(dropout_rate=0.5, activation='relu')
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'acc'])
    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[call_ES, ],
                        shuffle=True,
                        verbose=1)

    # plot_loss_acc(history, fold_ + 1)

    print('Loading Best Model')
    # # Get predicted probabilities for each class
    oof_preds[val_] = model.predict(x_valid, batch_size=batch_size)
    NN_predictions += model.predict(test_X, batch_size=batch_size) / folds.n_splits

print(NN_predictions)