# -*- coding: utf-8 -*-
"""Xception V1 model for Keras.

On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.

Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).

Also do note that this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers.

# Reference

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

import keras

from keras.models import Model


from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D,UpSampling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
import random



def Xception(input_shape=None, Ki=256, reduce=5, loop=8):

    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(reduce):

        prefix = 'block_reduce' + str(i)

        x = Conv2D(Ki, (3, 3), strides=(2, 2), use_bias=False, name=prefix+'conv1')(x)
        x = BatchNormalization(name=prefix+'conv1_bn')(x)
        x = Activation('relu', name=prefix+'conv1_act')(x)

    for i in range(loop):
        residual = x
        prefix = 'block_loop' + str(i)

        residual = Conv2D(Ki, (1, 1), use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name=prefix + '_conv1_act')(x)
        x = SeparableConv2D(Ki, (3, 3), padding='same', use_bias=False, name=prefix + '_conv1')(x)
        x = BatchNormalization(name=prefix + '_conv1_bn')(x)
        x = Activation('relu', name=prefix + '_conv2_act')(x)
        x = SeparableConv2D(Ki, (3, 3), padding='same', use_bias=False, name=prefix + '_conv2')(x)
        x = BatchNormalization(name=prefix + '_conv2_bn')(x)
        x = Activation('relu', name=prefix + '_conv3_act')(x)
        x = SeparableConv2D(Ki, (3, 3), padding='same', use_bias=False, name=prefix + '_conv3')(x)
        x = BatchNormalization(name=prefix + '_conv3_bn')(x)

        x = layers.add([x, residual])

    x = Activation('relu', name='block_out_acttemp')(x)
    x = SeparableConv2D(Ki*4, (3, 3), padding='same', use_bias=False, name='block_out_conv')(x)
    x = BatchNormalization(name='block_out_bn')(x)
    x = Activation('relu', name='block_out_act')(x)

    # Create model.
    model = Model(inputs, x, name='xception')

    return model
