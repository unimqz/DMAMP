import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from prettytable import PrettyTable
#from IPython.display import Image

from sklearn.preprocessing import LabelEncoder

# import tensorflow as tf

from keras.regularizers import l2
from keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from excel_read import excel_label, camp3_label, DBAASP_label
from utils import extract_data, extract_only_data

def read_data(data_path, partition):
  data = []
  for i in partition:
      content = open(os.path.join(data_path, i)).readlines()
      for con in content:
          if '>' not in con:
              data.append(con.strip())
  return np.array(data)




def plot_seq_count(df, data_name):
    sns.distplot(df)
    plt.title(f'Sequence count: {data_name}')
    plt.grid(True)

def data_construction(data_path, name_list, excel_name, pos_name_file):
    # reading all data_partitions
    all_data = read_data(data_path, name_list)
    all_label = np.concatenate((np.ones((int(0.5 * len(all_data)), 1)), np.zeros((int(0.5 * len(all_data)), 1))))

    second_label = excel_label(excel_name, data_path, pos_name_file)
    for i in range(second_label.shape[0]):
        temp = np.zeros(shape=(1, second_label.shape[1]))
        second_label = np.vstack([second_label, temp])
    # Split the data
    x_train, x_valid, y_train, y_valid = train_test_split(all_data, all_label, test_size=0.25, random_state=42,
                                                          shuffle=True, stratify=all_label)
    x_train1, x_valid1, y_train1, y_valid1 = train_test_split(all_data, second_label, test_size=0.25, random_state=42,
                                                          shuffle=True, stratify=all_label)
    #calculate class
    '''length = []
    for i in range(y_train1.shape[-1]):
        temp = 0
        for j in range(y_train1.shape[0]):
            if y_train1[j][i] == 1:
                temp+=1
        length.append(temp)
    print('train', length)

    length = []
    for i in range(y_valid1.shape[-1]):
        temp = 0
        for j in range(y_valid1.shape[0]):
            if y_valid1[j][i] == 1:
                temp+=1
        length.append(temp)
    print('val', length)'''

    # exit()

    # Length of sequence in train data.
    x_train_infor, x_valid_infor = {}, {}
    seq_len = []
    for x in x_train:
        seq_len.append(len(x))
    x_train_infor['seq_count'] = seq_len
    seq_len = []
    for x in x_valid:
        seq_len.append(len(x))
    x_valid_infor['seq_count'] = seq_len
    print(y_train1.shape)
    print(y_train.shape)
    x_train_infor['seq'] = list(x_train)
    x_valid_infor['seq'] = list(x_valid)
    x_train_infor['label1'] = list(y_train)
    x_valid_infor['label1'] = list(y_valid)
    #x_train_infor['label2'] = y_train1
    #x_valid_infor['label2'] = y_valid1

    x_train_infor['seq_count_max'] = max(x_train_infor['seq_count'])
    x_train_infor['seq_count_min'] = min(x_train_infor['seq_count'])
    x_valid_infor['seq_count_max'] = max(x_valid_infor['seq_count'])
    x_valid_infor['seq_count_min'] = min(x_valid_infor['seq_count'])
    print('Train: max:{}, min:{}\nValid: max:{}, min:{}'.format(x_train_infor['seq_count_max'],
                                                                x_train_infor['seq_count_min'],
                                                                x_valid_infor['seq_count_max'],
                                                                x_valid_infor['seq_count_min']))

    ##########plot seq distribution########
    plt.subplot(1, 2, 1)
    plot_seq_count(x_train_infor['seq_count'], 'Train')

    plt.subplot(1, 2, 2)
    plot_seq_count(x_valid_infor['seq_count'], 'Val')

    plt.subplots_adjust()
    #plt.show()

    ######构建dataframe######
    df_train = pd.DataFrame(x_train_infor)
    df_valid = pd.DataFrame(x_valid_infor)
    return df_train, df_valid, y_train1, y_valid1

def create_dict(codes):
    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index + 1
    return char_dict

def integer_encoding(data, char_dict):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
      and rest 4 are categorized as 0.
    """
    encode_list = []
    for row in data['seq'].values:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code, 0))
        encode_list.append(np.array(row_encode))

    return encode_list

def category(df_train, df_valid, y_train2, y_val2 , codes, max_length):
    char_dict = create_dict(codes)

    train_encode = integer_encoding(df_train, char_dict)
    val_encode = integer_encoding(df_valid, char_dict)

    # padding sequences
    train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
    val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')

    # One hot encoding of sequences
    train_one = to_categorical(train_pad)
    val_one = to_categorical(val_pad)

    le = LabelEncoder()  # sklearn.preprocessing.LabelEncoder,标准化标签
    y_train_label = le.fit_transform(
        df_train['label1'].astype('str'))  # 将label转换成对应的label数字。比如0，1，2，3，4. fit好像还强调是对应的有几种，label是几。
    y_val_label = le.transform(df_valid['label1'].astype('str'))  # 将label转换成对应的label数字。比如0，1，2，3，4

    # One hot encoding of outputs
    y_train = to_categorical(y_train_label)
    y_val = to_categorical(y_val_label)

    return train_one, val_one, y_train, y_val, y_train2, y_val2

def final_feature_task1(data_path, name_file, feature_number):
    #feature_number = [3] #只剩下pssm了，pssm排到第三个
    all_data, first_label = extract_data(data_path, name_file[0], name_file[1], feature_number)
    #second_label = excel_label(excel_name, data_path, pos_name_file)
    #for i in range(second_label.shape[0]):
        #temp = np.zeros(shape=(1, second_label.shape[1]))
        #second_label = np.vstack([second_label, temp])

    le = LabelEncoder()  # sklearn.preprocessing.LabelEncoder,标准化标签
    y_train_label = le.fit_transform(first_label.astype('str'))  # 将label转换成对应的label数字。比如0，1，2，3，4. fit好像还强调是对应的有几种，label是几。

    # One hot encoding of outputs
    y_train = to_categorical(y_train_label)
    return all_data, y_train, all_data.shape[-1], all_data.shape[-2]


def final_feature(data_path, name_file, feature_number, excel_name, pos_name_file):
    #feature_number = [3] #只剩下pssm了，pssm排到第三个
    #print(len(name_file))
    if len(name_file) == 1:
        all_data, first_label = extract_data(data_path, name_file[0], feature_number, None)
        #this is suitable for the test, where only positive data
    elif len(name_file) == 2:
        all_data, first_label = extract_data(data_path, name_file[0], feature_number, name_file[1])
    else:
        print('Error! The number of data_file exceeds to the required, please check!')
        exit()
    #all_data, first_label = extract_data(data_path, name_file[0], name_file[1], feature_number)
    second_label = excel_label(excel_name, data_path, pos_name_file) #(879, 7)
    # seqnamelist, second_label, seqlist = camp3_label(pos_name_file)  # for camp3, the label is in the header with '>'
    # seqnamelist, second_label, seqlist = DBAASP_label(pos_name_file)  # for DBAASP, the label is in the header with '>'
    # second_label = np.array(np.zeros(shape= (6, 7)))

    for i in range(first_label.shape[0]-second_label.shape[0]):
        temp = np.zeros(shape=(1, second_label.shape[1]))
        second_label = np.vstack([second_label, temp])


    le = LabelEncoder()  # sklearn.preprocessing.LabelEncoder,标准化标签
    y_train_label = le.fit_transform(first_label.astype('str'))  # 将label转换成对应的label数字。比如0，1，2，3，4. fit好像还强调是对应的有几种，label是几。

    # One hot encoding of outputs
    y_train = to_categorical(y_train_label)
    if len(name_file) == 1:
        y_train = np.hstack((y_train, np.zeros(shape=(y_train.shape[0], 1))))

    # y_train = np.array(y_train)
    # second_label = np.array(second_label)
    # print('%%%', type(second_label))
    #return all_data, y_train, second_label, all_data.shape[-1], all_data.shape[-2]
    return all_data, y_train, second_label

def final(data_path, name_file, seq_len, excel_name, pos_name_file):
    df_train, df_valid, y_train2, y_val2 = data_construction(data_path, name_file, excel_name, pos_name_file)

    #####################
    ###Text preprocessing
    #####################
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # codes2int = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    max_length = seq_len
    train_one, val_one, y_train1, y_val1, y_train2, y_val2 = category(df_train, df_valid, y_train2, y_val2, codes,
                                                                      max_length)
    plt.style.use('ggplot')
    return train_one, val_one, y_train1, y_val1, y_train2, y_val2, train_one.shape[-1]

def final_feature_label2(data_path, name_file, feature_number, excel_name, pos_name_file):
    all_data = extract_only_data(data_path, name_file[0], feature_number)
    second_label = excel_label(excel_name, data_path, pos_name_file) #(879, 7)
    return all_data, second_label


def __main__():
    final()