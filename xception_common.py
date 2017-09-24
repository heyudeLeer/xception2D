# -*- coding: utf-8 -*-
"""
commom model from Xception.

"""
from __future__ import print_function
from __future__ import absolute_import

import keras
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import BatchNormalization
from keras.models import Model


def Xception(input_shape=None, Ki=256, reduce=5, loop=8):

    inputs = Input(shape=input_shape)
    x = inputs

    #keep rgb atomic
    #x = Conv2D(1, (1, 1), use_bias=False, kernel_initializer=keras.initializers.Ones(), name='rgb2Atomic')(x)

    for i in range(reduce):
        prefix = 'block_reduce' + str(i)

        x = Conv2D(Ki, (3, 3), strides=(2, 2), use_bias=False, name=prefix+'_conv1')(x)
        x = BatchNormalization(name=prefix+'_conv1_bn')(x)
        x = Activation('relu', name=prefix+'_conv1_act')(x)

    for i in range(loop):
        prefix = 'block_loop' + str(i)

        residual = x
        x = SeparableConv2D(Ki, (3, 3), padding='same', use_bias=False, name=prefix + '_conv1')(x)
        x = BatchNormalization(name=prefix + '_conv1_bn')(x)
        x = Activation('relu', name=prefix + '_conv1_act')(x)
        x = SeparableConv2D(Ki, (3, 3), padding='same', use_bias=False, name=prefix + '_conv2')(x)
        x = BatchNormalization(name=prefix + '_conv2_bn')(x)
        x = Activation('relu', name=prefix + '_conv2_act')(x)
        x = SeparableConv2D(Ki, (3, 3), padding='same', use_bias=False, name=prefix + '_conv3')(x)
        x = BatchNormalization(name=prefix + '_conv3_bn')(x)
        x = Activation('relu', name=prefix + '_conv3_act')(x)

        x = layers.add([x, residual])

    x = SeparableConv2D(Ki*4, (3, 3), padding='same', use_bias=False, name='block_out_conv')(x)
    x = BatchNormalization(name='block_out_bn')(x)
    x = Activation('relu', name='block_out_act')(x)

    # Create model.
    model = Model(inputs, x, name='xception')

    return model
