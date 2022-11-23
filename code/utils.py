import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import (
    LeakyReLU,
    LSTM,
    GRU,
    SimpleRNN,
    Bidirectional,
)
from keras.regularizers import l2
from keras.callbacks import  Callback
from keras import backend as K
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score, precision_recall_fscore_support, roc_curve, classification_report
from sklearn.preprocessing import label_binarize

import tensorflow as tf

_EPSILON = 10e-8

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def display_model_score(model, train, val, batch_size):
    train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
    print('Train loss: ', train_score[0])
    print('Train accuracy: ', train_score[1])
    print('-' * 70)

    val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
    print('Val loss: ', val_score[0])
    print('Val accuracy: ', val_score[1])
    print('-' * 70)


def get_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state

def w_bin_xentropy(y_pred, y_true, label_weight):
    return np.mean(-(y_true * np.log(y_pred).clip(-1e10, 1e10) \
            + (1. - y_true) * np.log(1. - y_pred).clip(-1e10, 1e10)))

def weighted_binary_crossentropy(label_weight):
    def binary_crossentropy(y_true, y_pred):
        #weight = K.reshape(label_weight, (K.shape(label_weight)[1], -1))
        weight = label_weight
        return K.mean(weight * K.binary_crossentropy(
                                    output=y_pred, target=y_true), axis=-1)
    return binary_crossentropy

def ex_weighted_binary_crossentropy(label_weight, alpha, B, K):
    def binary_crossentropy(y_true, y_pred):
        #weight = K.reshape(label_weight, (K.shape(label_weight)[1], -1))
        weight = label_weight
        true_log = y_true * K.log(K.clip(y_true, K.epsilon(), None))
        pred_log = y_pred * K.log(K.clip(y_pred, K.epsilon(), None))

        reg = -alpha * (true_log + pred_log)
        for i in range(B):
            reg[:, i*K: (i+1)*K] * (K-i-1)
        reg = K.mean(reg, axis=-1)

        return reg + K.mean(weight * K.binary_crossentropy(
                                    output=y_pred, target=y_true), axis=-1)
    return binary_crossentropy

def reweight_with_scoring_fn(truth, pred, scoring_fn, b):
    for k in range(truth.shape[1]):
        t0 = np.copy(pred[:, j, :])
        t0[:, k] = 0
        t1 = np.copy(pred[:, j, :])
        t1[:, k] = 1

        weight[:, j, k] = \
            np.abs(self.scoring_fn(truth[:, j, :], t0) \
                   - self.scoring_fn(truth[:, j, :], t1))
    weight *= weight.size / weight.sum()
    return weight

def get_rnn_unit(rnn_unit, shape, inputs, l2w,
        activation='sigmoid', recurrent_dropout=0.5, **kwargs):
    regularizer = l2w

    if rnn_unit == 'simplernn':
        outputs = SimpleRNN(shape, return_sequences=True,
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=recurrent_dropout,
                activation=activation, **kwargs)(inputs)
    elif rnn_unit == 'lstm':
        outputs = LSTM(shape, return_sequences=True,
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=recurrent_dropout,
                activation=activation, **kwargs)(inputs)
    elif rnn_unit == 'gru':
        outputs = GRU(shape, return_sequences=True,
                recurrent_regularizer=regularizer,
                kernel_regularizer=regularizer,
                recurrent_dropout=recurrent_dropout,
                activation=activation, **kwargs)(inputs)
    else:
        raise NotImplementedError
    return outputs

def input_data(data, label, num_classes=5):
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    from keras.utils.np_utils import to_categorical

    labels = to_categorical(label, num_classes)
    #print(labels)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=True):
    while 1:  # 要无限循环
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        #print(indices)
        for start_idx in range(0, len(inputs), batch_size):
            batch_end = min(start_idx + batch_size, len(inputs))
            excerpt = indices[start_idx:batch_end]
            #print(start_idx, batch_end)
            return inputs[excerpt], targets[excerpt]

def read_data(data):
    file = open(data, 'rb')
    return pickle.load(file)

def batch_data(X):
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(X)

def extract_feature(feature_list, data):
    first_feature = data[feature_list[0]]
    #first_feature = batch_data(first_feature)
    num_samples = first_feature.shape[0]
    if len(feature_list) == 1 or (len(feature_list) == 1 and '9' in feature_list):
        return first_feature
    for ite in range(len(feature_list)):
        if ite == 0:
            continue
        feature = data[feature_list[ite]]
        feature = batch_data(feature)
        num_sample = feature.shape[0]
        if num_sample != num_samples:
            print('Error, feature shape not same...')
            exit()
        first_feature = np.concatenate((first_feature, feature), axis=1)
    return first_feature

def extract_data(data_path, pos_name, feature_number, neg_name):
    pos_data = os.path.join(data_path, pos_name)
    #print(pos_data)
    name_pos = read_data(pos_data)
    #print(name_pos.shape)
    pos_feature_Select = extract_feature(feature_number, name_pos)

    all_label = np.ones((pos_feature_Select.shape[0], 1))
    all_feature = pos_feature_Select
    if neg_name != None:
        neg_data = os.path.join(data_path, neg_name)
        name_neg = read_data(neg_data)  ##type:dict. dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        neg_feature_Select = extract_feature(feature_number, name_neg)
        all_label = np.concatenate(
            (all_label, np.zeros((neg_feature_Select.shape[0], 1))))
        all_feature = np.concatenate((pos_feature_Select, neg_feature_Select))

    #assert pos_feature_Select.shape[0] == neg_feature_Select.shape[0], 'postive and negative have different samples'
    return all_feature, all_label

def extract_only_data(data_path, pos_name, feature_number):
    pos_data = os.path.join(data_path, pos_name)
    print(pos_data)
    name_pos = read_data(pos_data)
    pos_feature_Select = extract_feature(feature_number, name_pos)
    return pos_feature_Select

def metric_function(x, y, model):
    y_one_hot = label_binarize(y, np.arange(2))
    y_score = model.predict(x)  # 输出的是整数标签
    mean_accuracy = model.score(x, y)
    print("mean_accuracy: ", mean_accuracy)
    print("predict label:", y_score)
    print(y_score == y)
    print(y_score.shape)
    y_score_pro = model.predict_proba(x)  # 输出概率
    print(y_score_pro)
    print(y_score_pro.shape)
    y_score_one_hot = label_binarize(y_score, np.arange(2))  # 这个函数的输入必须是整数的标签哦
    print(y_score_one_hot.shape)

    obj1 = confusion_matrix(y, y_score)  # 注意输入必须是整数型的，shape=(n_samples, )
    print('confusion_matrix\n', obj1)

    print(y)
    print('accuracy:{}'.format(accuracy_score(y, y_score)))  # 不存在average
    print('precision:{}'.format(precision_score(y, y_score, average='micro')))
    print('recall:{}'.format(recall_score(y, y_score, average='micro')))
    print('f1-score:{}'.format(f1_score(y, y_score, average='micro')))
    print('f1-score-for-each-class:{}'.format(precision_recall_fscore_support(y, y_score)))  # for macro
    # print('AUC y_pred = one-hot:{}\n'.format(roc_auc_score(y_one_hot, y_score_one_hot,average='micro')))  # 对于multi-class输入必须是proba，所以这种是错误的

    # AUC值
    auc = roc_auc_score(y_one_hot, y_score_pro, average='micro')  # 使用micro，会计算n_classes个roc曲线，再取平均
    print("AUC y_pred = proba:", auc)
    # 画ROC曲线
    print("one-hot label ravelled shape:", y_one_hot.ravel().shape)
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score_pro.ravel())  # ravel()表示平铺开来,因为输入的shape必须是(n_samples,)
    print("threshold： ", thresholds)
    plt.plot(fpr, tpr, linewidth=2, label='AUC=%.3f' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # 画一条y=x的直线，线条的颜色和类型
    plt.axis([0, 1.0, 0, 1.0])  # 限制坐标范围
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    # p-r曲线针对的是二分类，这里就不描述了

    ans = classification_report(y, y_score, digits=5)  # 小数点后保留5位有效数字
    print(ans)
