#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Train a 2D Xception model on the any small dataset.
'''


import  to2D_train
imgPath = "/home/heyude/PycharmProjects/data/dishes1.2/"


'''
creat model
'''
to2D_train.creatModel(modePar='out19', weights_h5=None, train=True)
#to2D_train.creatModel(modePar='common', weights_h5='common_09-07_00:13_0.940820443847.h5', train=False)
#to2D_train.creatModel(modePar='out37', weights_h5='out_37_09-01_16:07_0.993947545393.h5', train=False)
#to2D_train.creatModel(modePar='out19', weights_h5='out37_09-05_17:18_0.93110367893.h5', train=False)

#Basic api example
'''
Image segmentation
'''
#to2D_train.segImgfile(imgPath+'seg0720/5.jpg')

'''
Images segmentation from directory
'''
#to2D_train.segImgDir(imgPath+'seg')


#expanded api example
'''
predict imgs from directory
'''
#to2D_train.predictImgSets(imgPath+'0728Test', top=1)

'''
Calculate prediction accuracy by folder_name
#images path
        |
        |__folder_name
        |        |
                 |--imgs(The same class)
                 |
'''
to2D_train.CalcAccuracyImgDir(imgPath+'validation', top=1)

'''
show the 2D output for img predict Visualization
'''
#to2D_train.imgPredict2DShow(imgPath+'seg0720/1.jpg')

'''
Image location
'''
#to2D_train.locateImgfile(imgPath+'seg0720/5.jpg')



