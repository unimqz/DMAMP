class AMPNet(Model):
    def __init__(self):
        super(AMPNet, self).__init__()

    def resnet(self):
        x_input = Input(shape=(featute_input_length, 1))
        temp = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x_input)
        # print(temp.shape)
        temp = MaxPooling1D(pool_size=2, strides=2, padding='valid')(temp)
        # print(temp.shape)
        temp = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(temp)
        # print(temp.shape)
        temp = MaxPooling1D(pool_size=2, strides=2, padding='valid')(temp)
        # print(temp.shape)
        temp1 = self.residual_block(temp, 128, 1)
        # print(temp1.shape)
        temp1 = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(temp1)
        # print(temp.shape)
        temp1 = MaxPooling1D(pool_size=2, strides=2, padding='valid')(temp1)
        # print(temp.shape)
        temp1 = self.residual_block(temp1, 256, 2)
        # print(temp1.shape)
        # intertemp = temp1 #(None, 130, 256)
        temp = Dropout(0.35, name='dropout')(temp1)
        return Model(inputs=x_input, outputs=temp)

    def residual_block(self, data, filters, d_rate):
        """
        _data: input
        _filters: convolution filters
        _d_rate: dilation rate
        """
        shortcut = data
        # print('resblock, 1, ', data.shape)
        bn1 = BatchNormalization()(data)
        # print('bn1.shape', bn1.shape)
        act1 = Activation('relu')(bn1)
        conv1 = Conv1D(filters, 1, dilation_rate=d_rate, padding='valid', kernel_regularizer=l2(0.001))(act1)
        # print('resblock, 2, ', conv1.shape)
        # bottleneck convolution
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=l2(0.001))(act2)

        # skip connection
        # print('shortcut, ', shortcut.shape)
        # print('conv2, ', conv2.shape)
        x = Add()([conv2, shortcut])

        return x

    def task1(self):
        x_input1 = Input(shape=(130, 256))
        temp = Dense(units=64, activation='softmax', name='dropout1')(x_input1)
        # print(temp.shape)
        shape_dense = temp.shape[-1] * temp.shape[-2]
        temp = Reshape((-1, shape_dense))(temp)
        # print(temp.shape)
        temp = Dense(units=2, activation='softmax', name='dropout2')(temp)
        # print(temp.shape)
        temp = Flatten(name='one_output')(temp)
        # print(temp.shape)
        return Model(inputs=x_input1, outputs=temp)

    def task2(self):
        x_input1 = Input(shape=(130, 256))
        x_input2 = Input(shape=(featute_input_length, 1))
        print(x_input2.shape)
        feature_output = Reshape((-1, 8))(x_input2)
        print(x_input1.shape, feature_output.shape)
        temp = tf.concat([x_input1, feature_output], axis=-1)
        print(temp.shape)

        temp = Dense(units=64, activation='sigmoid', name='dropout1')(temp)
        print(temp.shape)
        shape_dense = temp.shape[-1] * temp.shape[-2]
        temp = Reshape((-1, shape_dense))(temp)
        print(temp.shape)
        temp = Dense(units=7, activation='sigmoid', name='dropout2')(temp)
        print('&&&&&&&&&&&', temp.shape)
        output = Flatten(name='second_output')(temp)
        return Model(inputs=[x_input1, x_input2], outputs=output)

    def build_model(self):
        x_input = Input(shape=(featute_input_length, 1))
        print('##########', x_input.shape) #(None, 2) (None, 6, 64)
        inter_output = self.resnet()(x_input)
        output1 = self.task1()(inter_output)
        output2 = self.task2()([inter_output, x_input])
        # print('output2,', output2.shape, 'output1,', output1.shape)
        return Model(inputs = x_input, outputs = {'one_output':output1, 'second_output':output2})
