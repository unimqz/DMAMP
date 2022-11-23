# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Constant, RandomNormal, RandomUniform
class BinaryFocalloss(keras.losses.Loss):
    """
    Implementation of focal loss. (<https://arxiv.org/pdf/1708.02002.pdf>)(Kaiming He)
    Binary Focal Loss Formula: FL = - y_true * alpha * (1-y_pred)^gamma * log(y_pred)
                             - (1 - y_true) * (1-alpha) * y_pred^gamma * log(1-y_pred)
                        ,which alpha = 0.25, gamma = 2, y_pred = sigmoid(x), y_true = target_tensor,
                        y_pred.shape = (batch_size, 1), y_true.shape = (batch_size, 1).
    """
    def __init__(self,
                 smoothing=0.0,
                 alpha=0.25,
                 gamma=2,
                 name='binary_focalloss',
                 **kwargs):
        """
        Initializes Binary Focal Loss class and sets attributes needed in loss calculation.
        :param smoothing: float, optional amount of label smoothing to apply. Set to 0 for no smoothing.
        :param alpha: float, optional amount of balance to apply (as in balanced cross entropy).
        :param gamma: int, optional amount of focal smoothing to apply.
                      Set to 0 for regular balanced cross entropy.
        :param name: str, optional name of this loss class (for tf.Keras.losses.Loss).
        :param kwargs: {'reduction': tf.keras.losses.Reduction.AUTO}
        """
        super(BinaryFocalloss, self).__init__(name = name, **kwargs)
        #print('smoothing vvv', smoothing) ##引用自定义loss的时候，一定要记得loss(),不是loss，
        assert smoothing <= 1 and smoothing >= 0, '`smoothing` needs to be in the range [0, 1].'
        assert alpha <= 1 and alpha >= 0, '`alpha` needs to be in the range [0, 1].'
        assert gamma >= 0, '`gamma` needs to be a non-negative integer.'
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """
        Computes binary focal loss between predicted probabilities and true label.
        :param y_true: ground truth labels. shape = (batch_size, 1)
        :param y_pred: predicted probabilities (softmax or sigmoid). shape = (batch_size, 1)
        :return: focal loss.
        """
        y_true = tf.cast(y_true, tf.float32)

        # 1\ Label Smoothing.
        if self.smoothing > 0:
            y_true = y_true * (1.0 - self.smoothing) + 0.5 * self.smoothing

        # 2\ Clip values for Numerical Stable. (Avoid NaN calculations)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 3\ Calculate the focal loss.
        focal = y_true * self.alpha * tf.pow((1-y_pred), self.gamma) * tf.math.log(y_pred)
        focal += (1 - y_true) * (1-self.alpha) * tf.pow(y_pred, self.gamma) * tf.math.log(1 - y_pred)
        loss = -focal

        # 4\ Sample Weight and Reduction
        # Note: sample_weight and reduction are implemented in the __call__ function.
        # In the super class tf.keras.losses.Loss, the __call__ function will invoke the call function.

        return loss

class weight_sample_loss(keras.losses.Loss):

    def __init__(self,
                 name='weight_sample_loss',
                 **kwargs):
        """
        Initializes Binary Focal Loss class and sets attributes needed in loss calculation.
        :param smoothing: float, optional amount of label smoothing to apply. Set to 0 for no smoothing.
        :param alpha: float, optional amount of balance to apply (as in balanced cross entropy).
        :param gamma: int, optional amount of focal smoothing to apply.
                      Set to 0 for regular balanced cross entropy.
        :param name: str, optional name of this loss class (for tf.Keras.losses.Loss).
        :param kwargs: {'reduction': tf.keras.losses.Reduction.AUTO}
        """
        super(weight_sample_loss, self).__init__(name = name, **kwargs)
        #print('smoothing vvv', smoothing) ##引用自定义loss的时候，一定要记得loss(),不是loss，

    def call(self, y_true, y_pred):
        """
        Computes binary focal loss between predicted probabilities and true label.
        :param y_true: ground truth labels. shape = (batch_size, 1)
        :param y_pred: predicted probabilities (softmax or sigmoid). shape = (batch_size, 1)
        :return: focal loss.
        """
        y_true = tf.cast(y_true, tf.float32)
        c_bool, pos_idx = self.match(y_true)
        # pos_idx = tf.py_function(func=self.match, inp=[y_true], Tout=[tf.float32])
        # tf.print(type(pos_idx))
        # print('%%%', pos_idx)
        # tensord = tf.fill((tf.shape(y_true)[0],),True)
        # tensord = tf.constant(True, shape=(tf.shape(y_true)[0],), dtype=bool)
        # rescc = tf.equal(tf.reduce_sum(tensord), tf.reduce_sum(pos_idx))
        # rescc = tf.cast(rescc, tf.int32)
        vv = tf.zeros(shape=(), dtype=tf.int32)
        if tf.equal(vv, c_bool) == True:
            # print('%%3', pos_idx) #all data in this batch is false,
            loss = tf.constant([0.0], dtype=tf.float32)
        else:
            # print('......%', pos_idx)
            new_batch_true_pos = tf.boolean_mask(y_true, axis=0, mask=pos_idx)
            new_batch_pred_neg = tf.boolean_mask(y_pred, axis=0, mask=pos_idx)
            loss = tf.keras.losses.BinaryCrossentropy()(new_batch_true_pos, new_batch_pred_neg)
        return loss

    @tf.function
    def match(self, y_true):
        neg_shape = tf.zeros([tf.shape(y_true)[0], 7], dtype=tf.float32)
        aa = tf.equal(y_true, neg_shape)
        # print(aa)
        aa_int = tf.cast(aa, tf.int32)
        # print(aa_int)
        result = tf.equal(tf.reduce_sum(aa_int, axis=-1), tf.reduce_sum(tf.ones_like(aa_int), axis=-1))
        # print(result) #(None, 1)
        result = tf.cast(result, tf.int32)
        # print(result)#(None, 1)
        v = tf.zeros(shape=(), dtype=tf.int32) #tf.Tensor(0, shape=(), dtype=int32)
        pos = tf.equal(result,v)
        print(pos.shape)
        c = tf.cast(pos, tf.int32)
        c= tf.reduce_sum(c)
        return c, pos
        # c = tf.equal(result, v)
        # c_int = tf.cast(c, tf.int32)
        # c_sum = tf.reduce_sum(aa_int)
        # print(c_sum)
        # ss = tf.py_function(self.euqal(c_sum, 1), [c_sum,1]
        # if ss:
        #     return 1, c
        # else:
        #     return 0, c

    @tf.function
    def euqal(self, a, b):
        if a == b:
            return 1
        else:
            return 0



import random as rn
import numpy as np
rand_seed = 42
np.random.seed(rand_seed)
rn.seed(rand_seed)
class multi_class(keras.layers.Layer):
    def __init__(self, batch_size,  **kwargs):
        super(multi_class, self).__init__()
        self.is_placeholder = True
        self.batch_size = batch_size
        #print('smoothing vvv', smoothing) ##引用自定义loss的时候，一定要记得loss(),不是loss，
        ######multi_loss = 1/(sigma1**2)*loss1+1/(sigma2**2)*loss2+log(sigma1**2)+log(sigma2**2)
        ######if sigma1 or sigma2 is zero, the loss will be nan.
        ######The latter can modified by log(sigma1**2+1), the previous can not be modified by this way.
        ######so i changed the trainable parameter to s, which K.exp(s) = sigma**2. In this way, the uncertain loss can be retain, and the nan
        ######will not be occur. followed by https://zhuanlan.zhihu.com/p/339537283  or ppaper: https://arxiv.org/pdf/2011.10671.pdf
        ######https://arxiv.org/pdf/1705.07115v3.pdf
        super(multi_class, self).__init__(**kwargs)

    def build(self, input_shape =None):
        self.log_sigma = []
        for i in range(2):
            self.log_sigma += [self.add_weight(name='uncertain_sigma'+str(i),
                                     shape=(1,),
                                     initializer= RandomNormal(seed=42),
                                     trainable = True)]
        super(multi_class, self).build(input_shape)

    def loss1(self, ys_true,ys_pred):
        loss11 = keras.losses.categorical_crossentropy(ys_true, ys_pred)
        # print(loss11.shape, '****************')
        self.batch_size = tf.shape(ys_true)[0]
        # print(self.batch_size)
        return tf.cast(K.sum(loss11, axis=0, keepdims=True), tf.float32)/tf.cast(self.batch_size, tf.float32)

    def loss2(self, ys_true,ys_pred):
        loss22 = weight_sample_loss()(ys_true, ys_pred)
        return loss22

    def multi_loss(self, ys1_true, ys2_true, ys1_pred, ys2_pred):
        loss1 = K.exp(-self.log_sigma[0])*self.loss1(ys1_true, ys1_pred)+0.5*(self.log_sigma[0])
        # print(self.log_sigma[0].shape, self.log_sigma[1].shape)
        # print(self.loss2(ys2_true, ys2_pred))
        loss2 = K.exp(-self.log_sigma[1])*self.loss2(ys2_true, ys2_pred)+0.5*(self.log_sigma[1])
        # print('loss1_multi_loss', loss1.shape)
        # print(loss2)
        # print(self.log_sigma[0], self.log_sigma[1])
        loss = loss1+loss2
        return loss, loss1, loss2

    def call(self, inputs):
        """
        Computes binary focal loss between predicted probabilities and true label.
        :param y_true: ground truth labels. shape = (batch_size, 1)
        :param y_pred: predicted probabilities (softmax or sigmoid). shape = (batch_size, 1)
        :return: focal loss.
        """
        assert isinstance(inputs, dict), print('Type Error! Custom Loss')
        ys1_true = inputs['y1_true']
        ys2_true = inputs['y2_true']
        ys1_pred = inputs['one_output']
        ys2_pred = inputs['second_output']
        # print('ys1_true', ys1_true.shape)
        # print('ys2_true', ys2_true.shape)
        # print('ys1_pred', ys1_pred.shape)
        # print('ys2_pred', ys2_pred.shape)

        loss, loss1, loss2= self.multi_loss(ys1_true, ys2_true, ys1_pred, ys2_pred)
        loss = tf.cast(loss, tf.float32)
        # print(loss)
        loss = tf.squeeze(loss)
        # print(loss.shape)
        #loss = tf.reduce_sum(loss)/self.batch_size
        self.add_loss(loss, inputs=inputs) #只是作为loss优化目标函数
        #print(loss1, loss2, loss)
        #print('loss', loss.shape)
        # print('total_loss', loss, loss1, loss2)
        return loss
