#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Train a 2D Xception model on any small dataset.

The following is an example:
Use xception to identify 24 dishes of a restaurant.
The dishes dataset path is imgPath(/home/heyude/PycharmProjects/data/dishes1.2/),images organized by training set and
verification set.
Thanks to keras ImageDataGenerator(),training set is very small, about 5 pictures per category,but verification set are
more than 50 pictures per category.
'''

import  to2D_train
imgPath = "/home/heyude/PycharmProjects/data/dishes1.2/"

#to2D_train.creatXceptionModel(mode=None, weights_h5=None, train=True)
#to2D_train.creatXceptionModel(mode=None, weights_h5='default_09-24_17:00_0.997982515131.h5', evaluate=False, train=True)
#to2D_train.creatXceptionModel(mode='out74', weights_h5='out74_09-01_20:34_0.98935462472.h5', train=True)
to2D_train.creatXceptionModel(mode='common', par=[512, 5, 18], weights_h5='common512_5_18_09-24_20:59_0.945150501672.h5', train=True)

'''
creat model for image recognition,location and segmentation

:param mode:None,change outsize or common
None:default xception mode from keras
change outsize:out10,out19,out37,out74;xception output will be change in order to improve pixel position accuracy
common:try a generic CNNs network

:param par:common mode subparameter
Ki:filters, Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).
reduce:Integer,Output size reduce factor
loop:Integer,Sub-network repeat times,determine the depth and entropy of the network

:param weights_h5:load pre-trained weights h5 file,File naming rules: mode + produce_time +evaluate_score

:param evaluate:boolean,Whether to evaluate this model

:param train:boolean,Whether to train this model

:return:created model
'''


#Once the model is created, you can test it with the following api

#segImgfile(url):
'''
image segmentation
:param file: image path
:return: print and plt show result
'''
to2D_train.segImgfile(imgPath+'seg0720/5.jpg')


#segImgDir(segPath):
'''
image segmentation in folder
:param segPath: images path
:return: print and plt show result
'''
to2D_train.segImgDir(imgPath+'seg')

#predictImgSets(setsPath,top=3):
'''
images prediction
:param setsPath: images path
:param top: The maximum probability of top categories
:return: img name, softmax rate,categories index
'''
to2D_train.predictImgSets(imgPath+'0728Test', top=1)


#CalcAccuracyImgDir(setsPath,top=3,verbose=1):
'''
calculate prediction accuracy by folder_name
:param setsPath: images path
:param top: The maximum probability of top categories
:param verbose: verbosity mode, 0 or 1.
:return:recognization accuracy
folder:
#images path
        |
        |__folder_name
        |        |
                 |--imgs(The same class)
                 |
'''
#to2D_train.CalcAccuracyImgDir(imgPath+'validation', top=1)


#imgPredict2DShow(url):
'''
img predict Visualization, segModel 2D output
:param url: image path
:return: plt show result
'''
to2D_train.imgPredict2DShow(imgPath+'seg0720/1.jpg')

#locateImgfile(url):
'''
Mark the location of the object with the rectangle of CV2
:param url: images path
:return: print and CV2 show image
'''
to2D_train.locateImgfile(imgPath+'seg0720/5.jpg')



