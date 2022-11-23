import numpy as np
import tensorflow as tf
import math
from sklearn import metrics
from tensorflow import keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, multilabel_confusion_matrix, auc, roc_curve\
    , average_precision_score, fbeta_score, jaccard_score, roc_auc_score, precision_recall_curve, classification_report, \
    coverage_error, label_ranking_average_precision_score, label_ranking_loss, hamming_loss, classification_report, precision_recall_fscore_support
import warnings

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def recall_metric(y_true, y_pred):
    """Recall metric.  micro_average
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_metric(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.  micro_average
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def cal_metric(y_true, y_pred):
    pass



def binary_metrics(y_true_ori, y_pred_ori):
    y_label = np.asarray(y_true_ori, dtype=np.int32)
    y_pred = np.asarray(y_pred_ori, dtype=np.int32)
    print(y_true_ori.shape, y_pred_ori.shape)
    assert y_true_ori.shape == y_pred_ori.shape, print('true and predict label should have same shape')
    y_pred = y_pred_ori[:, 0] #predict label one-hot 1-->[1,0]
    y_true = y_true_ori[:, 0] #true label one-hot 1-->[1,0]
    mm = confusion_matrix(y_true, np.rint(y_pred), labels=[1, 0])
    tp = mm[0][0]  #this index is correspond to labels in confusion_matrix
    fn = mm[0][1]
    fp = mm[1][0]
    tn = mm[1][1]
    precision = tp / (tp + fp)
    print('cal pre', precision)
    recall = tp / (tp + fn)  # sensitivity = reccall, tpr
    sp = tn / (tn + fp)
    acc = (tp+tn)/(tp+tn+fn+fp)
    acc_sco = accuracy_score(y_true, np.rint(y_pred))
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if mcc == 'nan':
        mcc = 0.0
    recall_sco = recall_score(y_true, np.rint(y_pred))
    precision_sco = precision_score(y_true, np.rint(y_pred))
    f1_sco = f1_score(y_true, np.rint(y_pred), labels=[1, 0])
    mcc_sco = matthews_corrcoef(y_true, np.rint(y_pred))
    precision_list ,recall_list, the = precision_recall_curve(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    auc_sco = auc(fpr,tpr)
    plot_roc(fpr, tpr, auc_sco, 'ROC')
    aupr = compute_aupr(precision_list ,recall_list)
    plot_roc(recall_list, precision_list, auc_sco, 'PR')
    print('acc: {:.3f}, metric.{:.3f}'.format(acc, acc_sco))
    print('precision: {:.3f}, metric.{:.3f}'.format(precision, precision_sco))
    print('mcc: {:.3f}, metric.{:.3f}'.format(mcc, mcc_sco))
    print('recall: {:.3f}, metric.{:.3f}'.format(recall, recall_sco))
    print('sp:{:.3f}, auc:{:.3f}, f1:{:.3f}, aupr:{:.3f}'.format(sp, auc_sco, f1_sco, aupr))
    return [recall_sco, sp, acc_sco, mcc_sco, auc_sco, f1_sco, precision_sco, aupr, tp, tn, fp, fn]

def compute_aupr(precision_list ,recall_list):
    auPR = metrics.auc(recall_list, precision_list)
    return auPR

def plot_roc(fpr, tpr, auc_sco, mode):
    """
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    name = ''
    if mode == 'ROC':
        name = 'AUC'
    elif mode == 'PR':
        name = 'AUPR'
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', label='%s (%s = %0.3f)' % (mode, name, auc_sco))
    plt.legend(loc='lower right')
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    save_name = name+'.png'
    plt.savefig(save_name, bbox_inches='tight')

def multi_metrics(y_true, y_pred):
    coverage = coverage_error(y_true, y_pred)
    ranking_ave_precision_score = label_ranking_average_precision_score(y_true, y_pred)
    ranking_loss = label_ranking_loss(y_true, y_pred)
    hl = hamming_loss(y_true, np.rint(y_pred))
    macro_precision_score = precision_score(y_true, np.rint(y_pred), average='macro')
    micro_precision_score = precision_score(y_true, np.rint(y_pred), average='micro')
    ave_acc = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, np.rint(y_pred))
    c_report = classification_report(y_true, np.rint(y_pred), target_names=['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6'])
    return [coverage, ranking_ave_precision_score, ranking_loss, hl, macro_precision_score, micro_precision_score, ave_acc, acc, c_report]

def simple_metrics(tp, fp, fn, tn):
    pred_sum = tp + fp
    label_sum = tp + fn
    precision = tp /pred_sum
    recall =tp / label_sum
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1_score = (1 + 1.0 ** 2) * precision * recall / (1.0 ** 2 * precision + recall)
    if np.isnan(f1_score):
        f1_score= 0.0

    #f1_score[np.isnan(f1_score)] = 0
    sp = tn / (tn + fp)
    mcc = (tp * fp - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (fp + fp) * (fp + fn)))
    if np.isnan(mcc):
        mcc = 0.0
    if np.isnan(acc):
        acc= 0.0
    if np.isnan(recall):
        recall= 0.0
    if np.isnan(precision):
        precision= 0.0
    if np.isnan(sp):
        sp= 0.0
    # return {'acc':acc, 'recall':recall, 'precision':precision, 'sp':sp, 'mcc':mcc}
    return [acc, recall, precision, sp, mcc, f1_score]

def get_multi_confusion_matrix(y_label, y_pred):

    y_label = np.array(y_label, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    # label_ind = np.where(y_label)
    labels = np.array([0, 1])
    # num classes
    num_labels = labels.size

    multi_cm = []
    # computer tp fp fn tn per class
    for l in range(num_labels):
        class_label = y_label == l
        class_pred = y_pred == l

        tp = np.sum(class_label * class_pred)
        fp = np.sum((1 - class_label) * class_pred)
        fn = np.sum(class_label * (1 - class_pred))
        tn = np.sum((1 - class_label) * (1 - class_pred))

        multi_cm.append([tn, fp, fn, tp])
    multi_cm = np.array(multi_cm).reshape(-1, 2, 2)
    return multi_cm

def multi_class_metrics(y_label, y_pred, beta=1.0, average='micro'):
    """
        :param y_label:
        :param y_pred:
        :param beta:
        :return:
        """
    y_label = np.asarray(y_label, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    # assert y_label.shape == y_pred.shape

    # ----------------------get confusion matrix of per class------------------------------
    num_class = y_label.shape[1]

    multi_cms = np.zeros((0, 2, 2))

    for i in range(num_class):
        multi_cm = get_multi_confusion_matrix(y_label[:, i], y_pred[:, i])
        multi_cms = np.concatenate([multi_cms, multi_cm[1][np.newaxis, :]])

    # ----------------------computer precision recall and f-score-------------------------
    tp = multi_cms[:, 1, 1]
    fp = multi_cms[:, 0, 1]
    fn = multi_cms[:, 1, 0]
    tn = multi_cms[:, 0, 0]

    tp_sum = tp
    pred_sum = tp + fp
    label_sum = tp + fn
    tn_sum = tn
    fp_sum = fp
    fn_sum = fn
    print(tp)
    #
    all_result_per_class = []
    for i in range(num_class):
        i_tp = tp[i]
        i_fp = fp[i]
        i_tn = tn[i]
        i_fn = fn[i]
        # print(i, i_tp, i_fp, i_fn, i_tn, '*')
        result  = simple_metrics(i_tp, i_fp, i_fn, i_tn)
        # print(result)
        all_result_per_class.append(result)
    ###calculate metrics of 'micro'
    if average == 'micro':
        tp_sum = np.array([tp.sum()])
        fp_sum = np.array([fp.sum()])
        tn_sum = np.array([tn.sum()])
        fn_sum = np.array([fn.sum()])
        pred_sum = np.array([pred_sum.sum()])
        label_sum = np.array([label_sum.sum()])

    precision = tp_sum / pred_sum

    # removing warnings if zero_division is set to something different than its default value
    warnings.filterwarnings('ignore')

    recall = tp_sum / label_sum
    print(tp_sum, tn_sum, fp_sum, fn_sum)
    acc = (tp_sum+tn_sum)/(tp_sum+tn_sum+fp_sum+fn_sum)
    f1_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    f1_score[np.isnan(f1_score)] = 0
    sp = tn_sum / (tn_sum + fp_sum)
    mcc = (tp_sum * fp_sum - fp_sum * fn_sum) / (
        np.sqrt((tp_sum + fp_sum) * (tp_sum + fn_sum) * (fp_sum + fp_sum) * (fp_sum + fn_sum)))
    if np.isnan(mcc):
        mcc = 0.0
    # print('mcc', mcc)
    # print('sp', sp)
    # print('acc', acc)

    precision = np.average(precision)
    recall = np.average(recall)
    f1_score = np.average(f1_score)
    mcc = np.average(mcc)
    spe = np.average(sp)
    acc = np.average(acc)
    if np.isnan(precision):
        precision = 0.0
    if np.isnan(recall):
        recall = 0.0
    if np.isnan(spe):
        spe = 0.0
    if np.isnan(acc):
        acc = 0.0
    return [acc, recall, precision, spe, mcc, f1_score], all_result_per_class

# y_pred = [[0.87,0.46,0.45,0.6], [0.3,0.6,0.4,0.3], [0.4,0.4,0.8,0.3]]
# y_true = [[1,0,1,0], [0,1,1,0], [0,0,1,1]]
def all_test_multi_metrcic(y_true, y_pred):
    confusion_m = get_multi_confusion_matrix(y_true, np.rint(y_pred))
    p1 = precision_score(y_true, np.rint(y_pred), average='micro')
    r1 = recall_score(y_true, np.rint(y_pred), average='micro')
    f1 = f1_score(y_true, np.rint(y_pred), average='micro')
    p2 = precision_score(y_true, np.rint(y_pred), average='macro')
    r2 = recall_score(y_true, np.rint(y_pred), average='macro')
    f2 = f1_score(y_true, np.rint(y_pred), average='macro')
    aps2 = average_precision_score(y_true, np.rint(y_pred), average='macro')
    aps1 = average_precision_score(y_true, np.rint(y_pred), average='micro')
    # print(p1, r1, f1, aps1)
    # print(p2, r2, f2, aps2)
    result = multi_class_metrics(y_true, np.rint(y_pred))  # acc, recall, precision, spe, mcc, f1_score
    # print(result, aps1, aps2)
    coverage = coverage_error(y_true, y_pred)
    ranking_ave_precision_score = label_ranking_average_precision_score(y_true, y_pred)
    ranking_loss = label_ranking_loss(y_true, y_pred)
    hl = hamming_loss(y_true, np.rint(y_pred))
    print('hamming loss:', hl, ' ranking loss:', ranking_loss, ' ranking_ave_precision_score:',
          ranking_ave_precision_score, ' coverage:', coverage)
    # print(confusion_m[-1])
    confu = multilabel_confusion_matrix(y_true, np.rint(y_pred))

    return result[0], aps1, result[1], aps2, [coverage, ranking_ave_precision_score, ranking_loss, hl], confu

multi_label_true = [[1, 1, 0, 0, 1],
                    [1, 0, 0, 1, 0],
                    [0, 1, 1, 0, 1]]

multi_label_pred = [[1, 0, 0, 1, 1],
                    [1, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1]]



'''
result = multi_metrics(multi_label_true, multi_label_pred)
result1= multi_class_metrics(multi_label_true, multi_label_pred)
print(result)
print(result1)
exit()'''


def calculate_auroc(predictions, labels):
    """
    Calculate auroc.
    :param predictions: predictions
    :param labels: labels
    :return: fpr_list, tpr_list, auroc
    """
    if np.max(labels) ==1 and np.min(labels)==0:
        fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
        auroc = metrics.roc_auc_score(labels, predictions)
    else:
        fpr_list, tpr_list = [], []
        auroc = np.nan
    return fpr_list, tpr_list, auroc


def calculate_aupr(predictions, labels):
    """
    Calculate aupr.
    :param predictions: predictions
    :param labels: labels
    :return: precision_list, recall_list, aupr
    """
    if np.max(labels) == 1 and np.min(labels) == 0:
        precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
        aupr = metrics.average_precision_score(labels, predictions)
    else:
        precision_list, recall_list = [], []
        aupr = np.nan
    return precision_list, recall_list, aupr

###################tensor metrics############################3
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)