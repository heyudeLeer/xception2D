
# -*- coding: utf-8 -*-
'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''
#!/usr/bin/python
# encoding: utf-8
#from __future__ import print_function
import keras
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, Input,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
import os
import scipy.misc
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import math
import numpy as np
from keras.applications import vgg16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
import matplotlib
import matplotlib.pyplot as plt

import uniout

'''
imgRows, imgCols = 299, 299
imgPath = "/home/heyude/PycharmProjects/data/dishes/"
imgsNumTrain=2162
imgsNumVal=1421
num_classes = 40
kernelSize = 8
fcnUpTimes = 37
'''
imgRows, imgCols = 299, 299
imgPath = "/home/heyude/PycharmProjects/data/dishes1.2/"
imgsNumTrain=122   #few shot
imgsNumVal=1503
trainTimes = 100
valTimes = 1
num_classes = 24

kernelSize = 10
fcnUpTimes = 30
#kernelSize = 11
#fcnUpTimes = 38

batchSize = 16


fitFlag = 0
trainFlag = 0
valFlag = 0

def get_session(gpu_fraction=0.8):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#KTF.set_session(get_session(0.8))

# this is the augmentation configuration we will use for training
datagen = ImageDataGenerator(
    #samplewise_center=True,
    #samplewise_std_normalization=True,
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    #zca_whitening=True,
    #zca_epsilon=1e-6,
    #rescale=1./255,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.4,
    zoom_range=0.4,
    channel_shift_range=3.0,
    horizontal_flip=True,
    vertical_flip=True,
    #fill_mode='nearest'
)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = datagen.flow_from_directory(
    imgPath + 'train',  # this is the target directory
    target_size=(imgRows, imgCols),  # all images will be resized to 150x150
    batch_size=batchSize,
    #save_to_dir=imgPath+'trainSet2', save_prefix='gen', save_format='png',
    shuffle=True,
    class_mode='categorical'
    )  # since we use binary_crossentropy loss, we need binary labels

#get train labs
classNameDic = train_generator.class_indices

# this is a similar generator, for validation data
validation_generator = datagen.flow_from_directory(
    imgPath + 'validation',
    target_size=(imgRows, imgCols),
    batch_size=batchSize,
    #save_to_dir=imgPath+'valSet2', save_prefix='gen', save_format='png',
    shuffle=True,
    class_mode='categorical'
)
'''
def creatTensor_SaveGenData2Disk(generator, imgNum, name):
    batches = 0
    tensorList = []
    for x_batch, _ in generator:
        print "batchs " + str(batches)
        tensorList.append(x_batch[0])
        batches += 1
        if batches >= imgNum:  # imgsNumTrain*4:  # math.ceil(imgsNumTrain/batchSize):
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    print 'augment batch finish'
    tensor_arry = np.array(tensorList)
    print tensor_arry.shape
    np.save(open(name, 'w'), tensor_arry)

    print(name + " Have Saved!")

    return tensor_arry

if not os.path.exists('train_source_img_tensorLib1_1.npy'):
    tensorArry = creatTensor_SaveGenData2Disk(train_generator_source, imgsNumTrain*trainTimes, 'train_source_img_tensorLib1_1.npy')
else:
    tensorArry = np.load(open('train_source_img_tensorLib1_1.npy'))

if not os.path.exists('val_source_img_tensorLib1_1.npy'):
    val_tensorArry = creatTensor_SaveGenData2Disk(val_generator_source, imgsNumVal*valTimes, 'val_source_img_tensorLib1_1.npy')
else:
    val_tensorArry = np.load(open('val_source_img_tensorLib1_1.npy'))
'''


# build the  network with ImageNet weights
inputShape = (imgRows, imgCols, 3)
base_model = Xception(weights='imagenet', include_top=False, input_shape=inputShape)
print('default Xception loaded.')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = Dense(512, activation='relu',name='fc1')(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have num_classes
predictions = Dense(num_classes, activation='softmax', name='fc')(x)

# this is the model we will train
FCmodel = Model(inputs=base_model.input, outputs=predictions, name='Xception_dishes')


fc = FCmodel.get_layer('fc')


# fcn
x = base_model.output
x = Dropout(0.5)(x)
#x = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='fcn')(x)
x = Conv2D(num_classes, (1, 1), activation='relu', padding='same', name='fcn')(x)         #hahaha da bug,relu
x = UpSampling2D((fcnUpTimes, fcnUpTimes))(x)
x = Activation('softmax')(x)
#x = Conv2DTranspose(10, (32,32))(x)
model = Model(inputs=base_model.input, outputs=x, name='Xception_dishes2Fcn')
fcn = model.get_layer('fcn')


from keras.models import load_model
if os.path.exists('XceptionweightsSize_fewShot-best.h5'):
    #model=load_model('Xception-2.h5')
    #print("...........load trained model......")
    print("...........load trained weights......")
    FCmodel.load_weights('XceptionweightsSize_fewShot-best.h5',by_name=True)
    FCmodel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True),metrics=['accuracy'])

    print
    print

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

if trainFlag==1:

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor= 'val_loss', patience=1) #val_loss

    # Let's train the model using RMSprop
    FCmodel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #for i, layer in enumerate(FCmodel.layers):
    #    print(i, layer.name)
    #exit(0)
    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    #for layer in FCmodel.layers[0:311]: #block5 top,mix8
    #    layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    #sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    #FCmodel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True), metrics=['accuracy'])
    #FCmodel.summary()
    # Fit the model on the batches generated by datagen.flow().
    FCmodel.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=math.ceil(imgsNumTrain*trainTimes/batchSize),
                    epochs=4,
                    callbacks=[early_stopping],
                    validation_data=validation_generator,
                    validation_steps=math.ceil(imgsNumVal*valTimes/batchSize),
                    workers=4)

    FCmodel.save_weights('XceptionweightsSize_fewShot-best.h5')

if valFlag:
    print 'evaluate...'
    score = FCmodel.evaluate_generator(validation_generator, steps=math.ceil(imgsNumVal/batchSize)*valTimes, workers=4)
    print(score)


def weight_fc2fcn(fc, fcn):

    weight = fc.get_weights()[0]
    print 'fc shape'
    print weight.shape

    #weight = weight[ : :-1]
    shape = fcn.get_weights()[0].shape
    array0 = weight.reshape(shape)         #25088 to (7,7,512),reshape就对？
    array1 = fcn.get_weights()[1]
    weight = [array0, array1]
    print('fcn shape')
    print(array0.shape)
    print(array1.shape)
    return weight

weights = weight_fc2fcn(fc, fcn)
fcn.set_weights(weights)
print
print 'fc weight to fcn done!'

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.summary()


#init color to
targetColor = np.array([
        (16, 1,  1),
        (1,  16, 1),
        (16, 16, 1),
        (1,  1,  16),
        (16, 1,  16),
        (1,  16, 16),
        (16, 16, 16),

        (32, 1,  1),
        (1,  32, 1),
        (32, 32,1),
        (1,  1,  32),
        (32, 1,  32),
        (1,  32, 32),
        (32, 32, 32),

        (48, 1,  1),
        (1,  48, 1),
        (48, 48, 1),
        (1,  1,  48),
        (48, 1,  48),
        (1,  48, 48),
        (48, 48,  48),

        (64, 1,  01),
        (0,  64, 01),
        (64, 46, 01),
    ])

# this is the augmentation configuration we will use for predict
predict_datagen = ImageDataGenerator(
    #samplewise_center=True,
    #samplewise_std_normalization=True,
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    #zca_whitening=True,
    #zca_epsilon=1e-6,
    #rescale=1./255,
)

#print 'start predict fit'
#predict_datagen.fit(tensorArry)

classNameDic_T = dict((v, k) for k, v in classNameDic.items())
print classNameDic_T

from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
def loadImage(url):

    img = [np.transpose(scipy.misc.imresize(scipy.misc.imread(url), (imgRows, imgCols)), (0, 1, 2))]
    img = np.array(img) #* 1.0 / 255
    #print img.shape #(1, 299, 299, 3)
    return img
    '''
    img_path = url
    img = image.load_img(img_path, target_size=(imgRows, imgCols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #print x.shape #(1, 299, 299, 3)
    return x
    '''
    for x_batch in predict_datagen.flow(x=img, y=None, batch_size=1):
        break
    return x_batch


def predictImage(url, fcn=False):
    img = loadImage(url)
    #img = [np.transpose(scipy.misc.imresize(scipy.misc.imread(url), (imgRows, imgCols)), (0, 1, 2))]
    #img = np.array(img)*1.0 / 255

    if fcn == True:
        preds = model.predict(img)
    else:
        preds = FCmodel.predict(img)
    return preds

#pred = predictImage(img_path)

#get top object in predict
#('name', 'degree','index')
def getTopObject(array, top=3):
    object = []
    arrayTemp = array[0][:]
    for i in range(top):
        index = arrayTemp.argmax()
        for className, classIndex in classNameDic.items():
            if index == classIndex and arrayTemp[index] > 0.0001:
                object.append((className, float('%.4f' % arrayTemp[index]), index))

        arrayTemp[index] = 0
    return object

#topObject = getTopObject(pred, 3)
#print topObject


def compuAcc(setsPath,top=3,verbose=1):
    totalNum=0
    totalAcc=0
    setsPath += '/'
    for _, dirs, _ in os.walk(setsPath):
        for dirName in dirs:
            print
            print ("coming "+ dirName)
            num = 0
            acc = 0
            for _, _, files in os.walk(setsPath+dirName):
                for name in files:
                    #print 'predict'+name
                    num += 1
                    totalNum +=1
                    pred = predictImage(setsPath+dirName+'/'+name)
                    topObject = getTopObject(pred, top+4)

                    for i in range(top):
                        ret=cmp(dirName, topObject[i][0]) #0:name
                        if ret==0:
                            acc += 1
                            totalAcc +=1
                            break
                        elif i==top-1 and verbose==1:
                            print (name+" distinguish err")
                            print (topObject)
                            break

            print (dirName+" accury is "+str(acc)+'/'+str(num))   #+str(float('%.4f' % (acc*1.0/num)))

    print()
    print ("total accury is " + str(float('%.4f' % (totalAcc*1.0/totalNum))))

#compuAcc(imgPath+'validation', verbose=1)


def testImgSets(setsPath,top=3):
    setsPath += '/'
    for _, dirs, _ in os.walk(setsPath):
        for dirName in dirs:
            print
            print ("coming "+ dirName)
            for _, _, files in os.walk(setsPath+dirName):
                for name in files:
                    print ('predict  '+name)
                    pred = predictImage(setsPath+dirName+'/'+name)
                    topObject = getTopObject(pred, top)
                    print (topObject)
                    print
predCorrectTh=0.1
tageCorrectTh=0.06
degreeCorrectTh=0.3
it_size = kernelSize*fcnUpTimes
overArea= it_size*it_size
def getRgbImgFromUpsampling(imgP):
    mylist = []
    matr = np.zeros((it_size, it_size, 3))
    for i in range(it_size):
        for j in range(it_size):
            #rgbIndex = imgP[0][i][j].argmax()
            rgbIndex = imgP[0, i, j, :].argmax()

            if imgP[0][i][j][rgbIndex] > predCorrectTh:

                matr[i, j, :] = targetColor[rgbIndex]
                mylist.append(rgbIndex)
                #if classNameDic_T.get(rgbIndex) not in dishNameList:
                #    dishNameList.append(classNameDic_T.get(rgbIndex))
                #    dishKeyList.append(rgbIndex)


    dishes_dictory = {}
    myset = set(mylist)  # myset是另外一个列表，里面的内容是mylist里面的无重复项
    for item in myset:

        key = item
        name = classNameDic_T.get(item)#item is dish index; key is dish name
        dot = mylist.count(item)   #value is dishes dot nums

        #get coordinate
        i_list=[]
        j_list=[]
        for i in range(it_size):
            for j in range(it_size):
                if matr[i, j, 0]== targetColor[item][0] and \
                   matr[i, j, 1]== targetColor[item][1] and \
                   matr[i, j, 2] == targetColor[item][2]:
                    i_list.append(i)
                    j_list.append(j)
        xl = min(j_list)
        yl = min(i_list)
        xr = max(j_list)
        yr = max(i_list)
        dishes_dictory[key] = (dot, name, (xl, yl, xr, yr))

    '''
    area =it_size*it_size
    for dish_k, dish_v in dishes_dictory.items():
        if dish_v[0] > area*0.05:
            print("maybe the %s has found %d" % (dish_v[1], dish_v[0]))
    '''

    return matr, dishes_dictory

#imgName = '/home/heyude/PycharmProjects/data/dishes/train/合利屋-鸡肉沙拉/IMG_4994.JPG'
#imgName = '/home/heyude/PycharmProjects/data/dishes/seg/ss1.jpg'
#imgName = '/home/heyude/PycharmProjects/data/dishes/seg/qs-fqjd-mf-lmsd4.jpg'
#imgName = '/home/heyude/PycharmProjects/data/dishes1.2/seg0720/1.jpg'
imgName = '/home/heyude/PycharmProjects/data/dishes/seg/qs-fqjd-mf1.jpg'

#imgP = predictImage(imgName)
#rgbImg = getRgbImgFromUpsampling(imgP)
testName= 'class'
def class2DShow(url):
    img = loadImage(url)
    # img = [np.transpose(scipy.misc.imresize(scipy.misc.imread(url), (imgRows, imgCols)), (0, 1, 2))]
    # img = np.array(img)*1.0 / 255

    #preds = FCmodel.predict(img)
    #print 'full pred..'
    #print preds

    preds2 = model.predict(img)
    #print preds2.shape
    #print preds2[0,:,:,0]


    n=num_classes/2
    plt.figure(figsize=(20, 10))
    for i in range(n):

        # display original
        ax = plt.subplot(2, n, i+1)
        ax.set_title(testName+'-'+str(i))
        ax.imshow(preds2[0, :, :, i])
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #ax.axis('off')

        # display reconstruction
        ax = plt.subplot(2, n, i+1 + n)
        ax.set_title(testName+'-'+str(i+n))
        ax.imshow(preds2[0, :, :, i+n])
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


'''
import cv2
#predImg(imgName)
pred = predictImage(imgName, fcn=True)
rgbImg, dish_info = getRgbImgFromUpsampling(pred)
bgrImg = rgbImg[:, :, ::-1]#rgb to bgr
cv2.imshow('bitmap', bgrImg)
img = loadImage(imgName)
print img.shape
#img = img[0][:, :, ::-1]#rgb to bgr
img = img[0]
#dish_info[key] = (dot, name, (xl, yl, xr, yr))
print img.shape
expand = 0
effective_p=0.6
for dish_k, dish_v in dish_info.items():
    xl, yl = dish_v[2][0:2]
    xr, yr = dish_v[2][2:4]
    if dish_v[0] > (xr-xl)*(yr-yl)*effective_p:
        print("the %s has found %d" % (dish_v[1], dish_v[0]))
        xl -= expand
        yl -= expand
        xr += expand
        yr += expand
        if xl<0:
            xl =0
        if yl<0:
            yl = 0
        if xr > imgCols:
            xr = imgCols
        if xr > imgRows:
            xr = imgRows
        print (xl,yl), (xr,yr)
        cv2.rectangle(img, (xl,yl), (xr,yr), (0, 255, 0), 2)
        cv2.putText(img,'dish-'+str(dish_k),(xl,yl),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        cv2.imshow(dish_v[1],img)
#cv2.waitKey()
#cv2.destroyAllWindows()
'''

zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
def segImgfile(file):

    plt.figure(figsize=(20, 5))

    print ('predict  '+file)
    img = loadImage(file)
    pred = model.predict(img)
    segImg,dishes_info = getRgbImgFromUpsampling(pred)

    # display source img
    ax = plt.subplot(2, 1, 1)
    ax.imshow(img[0])
    ax.set_title(file)
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 1, 2)
    for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
        xl = dish_v[2][0]
        yl = dish_v[2][1]
        xr = dish_v[2][2]
        yr = dish_v[2][3]
        h = xr - xl
        w=  yr - yl
        area = h * w
        tage = dish_v[0]*1.0/overArea
        degree = dish_v[0]*1.0/area
        if tage > tageCorrectTh and degree >degreeCorrectTh:
            if h>w*5 or w>h*5:
                break
            print("the %s mostly has found,degree:(%f,%f)" % (dish_v[1], tage,degree))
            central_x = (xr + xl)/2
            central_y = (yr + yl)/2
            #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
            ax.text(xl, central_y, (dish_v[1].decode('utf8'))[-4:0], size=8, color="r", fontproperties=zhfont1)

    #show result
    # display original
    ax.imshow(segImg)
    #ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()


def segImgDir(segPath):
    i=0
    for _, _, files in os.walk(segPath):
        print ("coming " + segPath)
        n = len(files)
        print 'files num is ' +str(n)
        plt.figure(figsize=(20, 5))
        for file in files:
            print
            print ('predict  '+file)
            img = loadImage(segPath + '/' + file)
            pred = model.predict(img)
            segImg,dishes_info = getRgbImgFromUpsampling(pred)

            i += 1
            # display source img
            ax = plt.subplot(2, n, i)
            ax.set_title(file[-8:-1])
            plt.imshow(img[0])

            # ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i+n)
            for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
                xl = dish_v[2][0]
                yl = dish_v[2][1]
                xr = dish_v[2][2]
                yr = dish_v[2][3]
                h = xr - xl
                w = yr - yl
                area = h * w
                tage = dish_v[0] * 1.0 / overArea
                degree = dish_v[0] * 1.0 / area
                if tage > tageCorrectTh and degree > degreeCorrectTh:
                    if h > w * 5 or w > h * 5:
                        break
                    print("the %s mostly has found,degree:(%f,%f)" % (dish_v[1], tage, degree))
                    central_x = (xr + xl) / 2
                    central_y = (yr + yl) / 2
                    #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
                    ax.text(xl, central_y, dish_v[1].decode('utf8'), size=8, color="r", fontproperties=zhfont1)

            # show result
            # display original
            #ax.set_title(file)
            ax.imshow(segImg)
            # ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


#testImgSets('/home/heyude/dishesLib/nas/a测试图片',top=5)

#testImgSets('/home/heyude/PycharmProjects/data/dishes/a测试图片',top=5)

#compuAcc(imgPath+'validation', top=1)
#compuAcc(imgPath+'dictoryTest', top=1)

#testImgSets(imgPath+'test', top=3)
#testImgSets(imgPath+'0728', top=1)
#testImgSets(imgPath+'0728Test', top=1)

class2DShow(imgPath+'temp/UNADJUSTEDNONRAW_thumb_29c4.jpg')
segImgfile(imgPath+'temp/UNADJUSTEDNONRAW_thumb_29c4.jpg')
#segImgDir(imgPath+'seg')
#segImgDir(imgPath+'seg0720')
#segImgDir(imgPath+'番茄鸡蛋0720')
#segImgDir(imgPath+'炒乌冬面0807')
#segImgDir(imgPath+'合利屋-炒乌冬面')