import numpy as np
import random
from keras.models import Model, Sequential, load_model, Input
from metrics_custom import binary_metrics, f1_metric, recall_metric, precision_metric, cal_metric, all_test_multi_metrcic
from loss_weight import weight_sample_loss
from data_preprocess_feature import final_feature
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

def save_result_test_task2(result_path, result_metric2, predict_result2, new_y_train2, title):
    #########save metric######
    fw_test = open(result_path + r'\{}.result'.format(title, ), 'w')
    fw_test.write(
        'acc\trecall\tprecision\tspe\tmcc\tf1_score\tAP\tmacro_acc\tmacro_recall\tmacro_precision\tmacro_spe\tmacro_mcc\tmacro_f1\tmacro_AP\tcoverage\tranking_ave_precision_score\trabking_loss\thanming_loss\n')
    for j in range(6):
        fw_test.write(str(result_metric2[0][j]) + '\t')
    fw_test.write(str(result_metric2[1]) + '\t')
    resu2 = np.array(result_metric2[2])
    acc1, recall1, precision1, spe1, mcc1, f1_score1 = np.mean(resu2[:, 0]), np.mean(resu2[:, 1]), np.mean(
        resu2[:, 2]), np.mean(resu2[:, 3]), np.mean(resu2[:, 4]), np.mean(resu2[:, 5])
    result22 = [acc1, recall1, precision1, spe1, mcc1, f1_score1]
    for j in range(6):
        fw_test.write(str(result22[j]) + '\t')
    fw_test.write(str(result_metric2[3]) + '\t')
    for j in range(4):
        fw_test.write(str(result_metric2[4][j]) + '\t')
    fw_test.write('\n\nthe metrics for per class ******\n')
    fw_test.write('acc\trecall\tprecision\tspe\tmcc\tf1_score\n')
    for i in range(resu2.shape[0]):
        for jind in range(resu2.shape[1]):
            fw_test.write(str(resu2[i][jind]) + '\t') #acc, recall, precision, sp, mcc, f1_score
        fw_test.write('\n')
    # confusion matrix shape (7, 2, 2)
    fw_test.write('\n\nconfusion matrix******\n')
    fw_test.write('class\tTN\tFP\tPN\tTP\n')
    for i in range(result_metric2[5].shape[0]):
        fw_test.write('class '+str(i)+'\t')
        for jind in range(result_metric2[5].shape[1]):
            for kind in range(result_metric2[5].shape[2]):
                fw_test.write(str(result_metric2[5][i][jind][kind]) + '\t')
        fw_test.write('\n')
    fw_test.close()

    #####save result######
    fw_test = open(
        result_path + r'\{}_predict.txt'.format(title), 'w')
    for i in range(predict_result2.shape[0]):
        for jind in range(predict_result2.shape[1]):
            fw_test.write(str(predict_result2[i][jind]) + '\t')
        fw_test.write('\n')
    fw_test.close()

    fw_test = open(
        result_path + r'\{}_true.txt'.format(title, ), 'w')
    for i in range(new_y_train2.shape[0]):
        for j in range(new_y_train2.shape[1]):
            fw_test.write(str(new_y_train2[i][j]) + '\t')
        fw_test.write('\n')
    fw_test.close()

def save_result_test_task1(y_test1, predict_result, result_path, title):
    fw = open(result_path+r'\{}.result'.format(title), 'w')
    result1 = binary_metrics(y_test1, predict_result)
    fw.write('acc' + '\t' + 'precision' + '\t' + 'sp' + '\t' + 'recall' + '\t' + 'mcc' + '\t' + 'auc_score' + '\t' + 'aupr' + '\t' + 'f1_score\n')
    fw.write(str(result1[2]) + '\t' + str(result1[6]) + '\t' + str(result1[1]) + '\t' + str(
            result1[0]) + '\t' + str(result1[3]) + '\t' + str(result1[4]) + '\t' + str(result1[7]) + '\t' + str(
            result1[5]) + '\n')
    fw.write(str(result1[8])+ '\t'+ str(result1[9])+ '\t'+str(result1[10])+ '\t'+str(result1[11])+'\n')
    fw.close()


#data preprocessing
data_path = r'Data'
name_file = ['TE226.pickle']
excel_name = r'Data\AMP_class_infor.xlsx'
pos_name_file = ['Data\\TE226.fasta']
feature_number= [6] #0:PSE_pssm 1:PSE_pp 2:PSE_AAC 3:AVB_pssm 4:AVB_pp 5:AVB_AAC  6:DWT_pssm 7:DWT_pp 8:DWT_AAC
test_data, y_test1, y_test2 = final_feature(data_path, name_file, feature_number, excel_name, pos_name_file)
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

featute_input_length = test_data.shape[1]

######load model
model_dir = 'model_pretrain\\final_model_710_epoch20_lr0.0001_bc10_task11_2'
model_combine = load_model(model_dir, custom_objects={'weight_sample_loss': weight_sample_loss, 'f1_metric':f1_metric, 'recall_metric': recall_metric, 'precision_metric': precision_metric})
print(model_combine.summary())
output = model_combine.predict(test_data)
print(output)


#save result
result_path = r'result_path'
if os.path.exists(result_path) == False:
    os.mkdir(result_path)
title_test = 'predict_task1_TE226'
save_result_test_task1(y_test1, output['one_output'], result_path, title_test)
fw_test = open(result_path + r'\{}_predict.txt'.format(title_test), 'w')
for i in range(output['one_output'].shape[0]):
    for jind in range(output['one_output'].shape[1]):
        fw_test.write(str(output['one_output'][i][jind]) + '\t')
    fw_test.write('\n')
fw_test.close()

title_test = 'predict_task2_TE226'
result_metric2 = all_test_multi_metrcic(y_test2, output['second_output'])
save_result_test_task2(result_path, result_metric2, output['second_output'], y_test2, title_test)
print('Finished.....')