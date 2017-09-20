
# encoding: utf-8

import os
import math
import time
import keras

from keras.models import Model
from keras.layers import Dropout, Activation, Input,GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import uniout
import cv2

'''
read_only global variabl
'''

num_classes = 24
imgRows = 299
imgCols = 299

imgsNumTrain=122   #few shot
imgsNumVal=1503
trainTimes = 120
valTimes = 1


judgeThreshold=0.05
degreeCorrectTh=0.3
tageCorrectTh=0.05

imgPath = "/home/heyude/PycharmProjects/data/dishes1.2/"


'''
Auto and global variabl, Writable
'''

kernelSize = 0
fcnUpTimes = 0
it_size = 0
overArea= 0
batchSize = 16
classNameDic=None
classNameDic_T=None
trainModel=None
segModel=None


def get_session(gpu_fraction=0.8):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(get_session(0.8))

# this is the augmentation configuration we will use for training
datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.4,
    zoom_range=0.4,
    channel_shift_range=3.0,
    horizontal_flip=True,
    vertical_flip=True,
    # fill_mode='nearest'
)


def creatModel(modePar='base', weights_h5=None, train=False):
    '''
    creat model for image recognition,location and segmentation
    :param modePar:End output size, base or out_37, base size is 10;
    :param weights_h5:load trained weights h5 file,file name format: modePar + produce_time +evaluate_score
    :param train:boolean,Whether to train this model
    :return:trained model
    '''
    global fcnUpTimes
    global kernelSize
    global trainModel
    global segModel
    global it_size
    global overArea
    global batchSize
    global classNameDic
    global classNameDic_T


    if modePar=='base':
        from keras.applications.xception import Xception
        kernelSize = 10
        fcnUpTimes = 30

    else:
        import xception_outsize_change
        from xception_outsize_change import Xception
        #from xception_common import Xception
        if modePar == 'common':
            kernelSize = 10
            fcnUpTimes = 30
            batchSize = 8  # GPU memory limit
            print 'you chose common mode...'
        if modePar == 'out19':
            xception_outsize_change.out19 = 1
            kernelSize = 19
            fcnUpTimes = 16
            print 'you chose out19 mode...'
        if modePar == 'out37':
            xception_outsize_change.out19 = 1
            xception_outsize_change.out37 = 1
            kernelSize = 37
            fcnUpTimes = 8
            batchSize = 8   #GPU memory limit
            print 'you chose out37 mode...'
        if modePar == 'out74':
            xception_outsize_change.out19 = 1
            xception_outsize_change.out37 = 1
            xception_outsize_change.out74 = 1
            kernelSize = 74
            fcnUpTimes = 4
            batchSize = 3   #GPU memory limit
            print 'you chose out74 mode...'

    it_size = min(kernelSize*fcnUpTimes, imgRows, imgCols)
    print "Pixel iteration size is "+str(it_size)
    overArea = it_size * it_size

    # build the network with ImageNet weights
    inputShape = (imgRows, imgCols, 3)
    #base_model = Xception(include_top=False, input_shape=inputShape)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=inputShape)
    #base_model = Xception(Ki=512, reduce=4, loop=32, input_shape=inputShape)
    print('base_model have loaded...')

    x = base_model.output
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, (1, 1), name='fcn')(x)
    y = x
    #2D to point
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    trainModel = Model(inputs=base_model.input, outputs=x, name='Xception_dishes')

    #   segmentation
    y = Activation('softmax')(y)
    #y = keras.layers.advanced_activations.ThresholdedReLU(theta=judgeThreshold)(y)
    y = UpSampling2D((fcnUpTimes, fcnUpTimes))(y)
    segModel = Model(inputs=base_model.input, outputs=y, name='Xception_dishes2Fcn')

    if weights_h5 != None and os.path.exists(weights_h5):
        print(weights_h5 +" have loaded!")
        trainModel.load_weights(weights_h5,by_name=True)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(trainModel.layers):
    #    print(i, layer.name)
    # exit(0)
    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for layer in trainModel.layers[0:311]: #block5 top,mix8
    #    layer.trainable = False
    #rgbTo1_layer = trainModel.get_layer(name='rgbTo1')
    #rgbTo1_layer.trainable = False

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    # sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = keras.optimizers.SGD(lr=1e-7, decay=1e-9, momentum=0.9, nesterov=True)
    #trainModel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-7, momentum=0.9, nesterov=True), metrics=['accuracy'])
    # Let's train the model using RMSprop
    trainModel.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    trainModel.summary()

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = datagen.flow_from_directory(
        imgPath + 'train',  # this is the target directory
        target_size=(imgRows, imgCols),  # all images will be resized to 150x150
        batch_size=batchSize,
        # save_to_dir=imgPath+'trainSet', save_prefix='gen', save_format='png',
        shuffle=True,
        class_mode='categorical'
    )  # since we use binary_crossentropy loss, we need binary labels

    # get train labs and item name
    classNameDic = train_generator.class_indices
    classNameDic_T = dict((v, k) for k, v in classNameDic.items())
    print classNameDic_T

    # this is a similar generator, for validation data
    validation_generator = datagen.flow_from_directory(
        imgPath + 'validation',
        target_size=(imgRows, imgCols),
        batch_size=batchSize,
        # save_to_dir=imgPath+'valSet', save_prefix='gen', save_format='png',
        shuffle=True,
        class_mode='categorical'
    )


    if train==True:
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        # Fit the model on the batches generated by datagen.flow().
        trainModel.fit_generator(
            generator=train_generator,
            steps_per_epoch=math.ceil(num_classes * trainTimes / batchSize),
            epochs=3,
            callbacks=[early_stopping],
            validation_data=validation_generator,
            validation_steps=math.ceil(imgsNumVal*valTimes/batchSize),
            workers=4)

        print 'evaluate...'
        score = trainModel.evaluate_generator(validation_generator, steps=math.ceil(imgsNumVal / batchSize) * valTimes, workers=4)
        print(score)
        timeInfo = time.strftime('%m-%d_%H:%M', time.localtime(time.time()))
        print timeInfo
        trainModel.save_weights(modePar+'_'+timeInfo+'_'+str(score[1])+'.h5')

    return trainModel, segModel

#RGB item for Image segmentation
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
for i in range(len(targetColor)):
        targetColor[i] += [128, 128, 128]

def loadImage(url):

    img = [np.transpose(scipy.misc.imresize(scipy.misc.imread(url), (imgRows, imgCols)), (0, 1, 2))]
    img = np.array(img) #* 1.0 / 255
    #print img.shape #(1, 299, 299, 3)
    return img

    '''
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import preprocess_input
    img_path = url
    img = image.load_img(img_path, target_size=(imgRows, imgCols))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    #print x.shape #(1, 299, 299, 3)
    return x
    '''

    for x_batch in predict_datagen.flow(x=img, y=None, batch_size=1):
        break
    return x_batch


def predictImage(url, fcn=False):
    img = loadImage(url)

    if segModel == None or trainModel == None:
        print 'please creatModel 1st'
        print 'use to2D_train.creatModel() function'
        return

    if fcn == True:
        preds = segModel.predict(img)
    else:
        preds = trainModel.predict(img)
    return preds


def getTopObject(array, top=3):
    '''
    get top object in predict
    :param array: model predict result
    :param top: The maximum probability of top categories
    :return: object:['name', 'degree','index']
    '''
    object = []
    arrayTemp = array[0][:]
    for i in range(top):
        index = arrayTemp.argmax()
        for className, classIndex in classNameDic.items():
            if index == classIndex and arrayTemp[index] > 0.0001:
                object.append((className, float('%.4f' % arrayTemp[index]), index))

        arrayTemp[index] = 0
    return object

def predictImgSets(setsPath,top=3):
    '''
    images prediction
    :param setsPath: images path
    :param top: The maximum probability of top categories
    :return: img name, softmax rate,categories index
    '''
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

def CalcAccuracyImgDir(setsPath,top=3,verbose=1):
    '''
    :param setsPath: images path
    :param top: The maximum probability of top categories
    :param verbose: verbosity mode, 0 or 1.
    :return:recognization accuracy
    '''
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


def getRgbImgFromUpsampling(imgP):
    '''
    Summarize the segModel results
    :param imgP: segModel predict
    :return:rgbImg indicate the categories,location; dishes_dictory:(category pixels, name, coordinate(xl, yl, xr, yr))
    '''
    mylist = []
    rgbImg = np.zeros((it_size, it_size, 3), dtype='uint8')
    for i in range(it_size):
        for j in range(it_size):
            rgbIndex = imgP[0, i, j, :].argmax()

            if imgP[0][i][j][rgbIndex] > judgeThreshold:
                rgbImg[i, j, :] = targetColor[rgbIndex]
                mylist.append(rgbIndex)

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
                if rgbImg[i, j, 0] == targetColor[item][0] and   \
                   rgbImg[i, j, 1] == targetColor[item][1] and  \
                   rgbImg[i, j, 2] == targetColor[item][2]:
                    i_list.append(i)
                    j_list.append(j)
        xl = min(j_list)
        yl = min(i_list)
        xr = max(j_list)
        yr = max(i_list)
        dishes_dictory[key] = (dot, name, (xl, yl, xr, yr))

    return rgbImg, dishes_dictory


def imgPredict2DShow(url):
    '''
    show the 2D output for predict Visualization
    :param url: image path
    :return: plt show result
    '''
    title = 'fea_map'
    img = loadImage(url)

    segmentation_map = segModel.predict(img)

    n=num_classes/2
    plt.figure(figsize=(20, 5))
    for i in range(n):

        ax = plt.subplot(2, n, i+1)
        ax.set_title(title+str(i))
        ax.imshow(segmentation_map[0, :, :, i])
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i+1 + n)
        ax.set_title(title+str(i+n))
        ax.imshow(segmentation_map[0, :, :, i+n])
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
def segImgfile(url):
    '''
    image segmentation
    :param file: image path
    :return: print and plt show result
    '''

    plt.figure(figsize=(20, 5))

    print ('predict  '+url)
    img = loadImage(url)
    pred = segModel.predict(img)
    print pred.shape
    segImg,dishes_info = getRgbImgFromUpsampling(pred)

    # display source img
    ax = plt.subplot(2, 1, 1)
    ax.imshow(img[0])
    ax.set_title(url)
    # ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 1, 2)
    foundSthFlag = 0
    for dish_k, dish_v in dishes_info.items():  # dish_v:(dot, name, (xl, yl, xr, yr))
        xl = dish_v[2][0]
        yl = dish_v[2][1]
        xr = dish_v[2][2]
        yr = dish_v[2][3]
        h = xr - xl
        w=  yr - yl
        area = h * w
        areaRate = dish_v[0]*1.0/overArea
        fullness = dish_v[0]*1.0/area
        if areaRate > tageCorrectTh and fullness >degreeCorrectTh:
            if h>w*8 or w>h*8:
                break
            #recognize something
            foundSthFlag = 1
            print("the %s mostly has found, AreaRate and fullness:(%f,%f)" % (dish_v[1], areaRate, fullness))
            central_x = (xr + xl)/2
            central_y = (yr + yl)/2
            #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
            ax.text(xl, central_y, (dish_v[1].decode('utf8'))[-4:0], size=8, color="r", fontproperties=zhfont1)
    if foundSthFlag:
        # display result
        ax.imshow(segImg)
        #ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def segImgDir(segPath):
    '''
    images in folder segmentation
    :param segPath: images path
    :return: print and plt show result
    '''
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
            pred = segModel.predict(img)
            segImg,dishes_info = getRgbImgFromUpsampling(pred)

            i += 1
            # display source img
            ax = plt.subplot(2, n, i)
            ax.set_title(file[-8:-1])
            plt.imshow(img[0])

            # ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i+n)
            foundSthFlag = 0
            for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
                xl = dish_v[2][0]
                yl = dish_v[2][1]
                xr = dish_v[2][2]
                yr = dish_v[2][3]
                h = xr - xl
                w = yr - yl
                area = h * w
                areaRate = dish_v[0] * 1.0 / overArea
                fullness = dish_v[0] * 1.0 / area
                if areaRate > tageCorrectTh and fullness > degreeCorrectTh:
                    if h > w * 8 or w > h * 8:
                        break
                    foundSthFlag = 1
                    print("the %s mostly has found,AreaRate and fullness:(%f,%f)" % (dish_v[1], areaRate, fullness))
                    central_x = (xr + xl) / 2
                    central_y = (yr + yl) / 2
                    #ax.text(central_x, central_y, dish_v[1].decode('utf8'), size=10, color="r", fontproperties=zhfont1)
                    #ax.text(xl, central_y, dish_v[1].decode('utf8'), size=8, color="r", fontproperties=zhfont1)

            if foundSthFlag:
                # display result
                ax.imshow(segImg)
                # ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.show()

def locateImgfile(url):

    print ('predict  ' + url)
    img = loadImage(url)
    pred = segModel.predict(img)
    segImg, dishes_info = getRgbImgFromUpsampling(pred)
    bgrImg = segImg[:, :, ::-1]  # rgb to bgr
    cv2.imshow('category pixels', bgrImg)

    img = img[0]
    for dish_k, dish_v in dishes_info.items():  # dish_v:(dot, name, (xl, yl, xr, yr))
        xl = dish_v[2][0]
        yl = dish_v[2][1]
        xr = dish_v[2][2]
        yr = dish_v[2][3]
        h = xr - xl
        w = yr - yl
        area = h * w
        areaRate = dish_v[0] * 1.0 / overArea
        fullness = dish_v[0] * 1.0 / area
        if areaRate > tageCorrectTh and fullness > degreeCorrectTh:
            if h > w * 8 or w > h * 8:
                break
            # recognize something
            print("the %s mostly has found, AreaRate and fullness:(%f,%f)" % (dish_v[1], areaRate, fullness))
            central_x = (xr + xl) / 2
            central_y = (yr + yl) / 2
            cv2.rectangle(img, (xl, yl), (xr, yr), (0, 255, 0), 2)
            cv2.putText(img, 'dish-' + str(dish_k), (central_x, central_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))

    img = img[:, :, ::-1]  # rgb to bgr
    cv2.imshow('location', img)
    cv2.waitKey()
    cv2.destroyAllWindows()