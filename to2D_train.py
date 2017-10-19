
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
imgRows = 369
imgCols = 369

imgsNumTrain = 149   #few shot
imgsNumVal = 1455
trainTimes = 10
valTimes = 1

judgeThreshold=0.5 #predict rate
tageCorrectTh=0.07 #area rate

degreeCorrectTh=0.55 #fullness

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
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    shear_range=0.4,
    zoom_range=0.4,
    #channel_shift_range=0.5,
    horizontal_flip=True,
    vertical_flip=True,
    # fill_mode='nearest'
)


def creatXceptionModel(mode=None, par=None, weights_h5=None, evaluate=False, train=False):
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
    global fcnUpTimes
    global kernelSize
    global trainModel
    global segModel
    global it_size
    global overArea
    global batchSize
    global classNameDic
    global classNameDic_T

    # build the network with ImageNet weights
    inputShape = (imgRows, imgCols, 3)

    # default
    if mode==None:
        from keras.applications.xception import Xception
        base_model = Xception(weights='imagenet', include_top=False, input_shape=inputShape)
        print 'you chose default mode'
        mode='default'
        kernelSize = 10
        fcnUpTimes = 30

    elif mode == 'common':
        from xception_common import Xception
        if par==None:
            print "common mode par should not be None!"
            exit(0)
        base_model = Xception(Ki=par[0], reduce=par[1], loop=par[2], input_shape=inputShape)
        parStr = str(par[0]) + '_' + str(par[1]) + '_' + str(par[2])
        mode += parStr
        print 'you chose common mode,par is ' + parStr
        kernelSize = 8  # need adapter with reduce
        fcnUpTimes = 37  # need adapter with reduce
        batchSize = 8  # GPU memory limit
    # outsize changed mode
    else:
        import xception_outsize_change
        from xception_outsize_change import Xception
        if mode == 'out10':# as same as default mode
            kernelSize = 10
            fcnUpTimes = 30
            print 'you chose out10 mode...'
        if mode == 'out19':
            xception_outsize_change.out19 = 1
            kernelSize = 19
            fcnUpTimes = 16
            print 'you chose out19 mode...'
        if mode == 'out37':
            xception_outsize_change.out19 = 1
            xception_outsize_change.out37 = 1
            kernelSize = 37
            kernelSize = 18
            fcnUpTimes = 8
            batchSize = 8   # GPU memory limit
            print 'you chose out37 mode...'
        if mode == 'out74':
            xception_outsize_change.out19 = 1
            xception_outsize_change.out37 = 1
            xception_outsize_change.out74 = 1
            kernelSize = 74
            fcnUpTimes = 4
            batchSize = 3   # GPU memory limit
            print 'you chose out74 mode...'

        # base_model = Xception(include_top=False, input_shape=inputShape)
        base_model = Xception(weights='imagenet', include_top=False, input_shape=inputShape)

    #it_size = min(kernelSize*fcnUpTimes, imgRows, imgCols)
    #print "Pixel iteration size is "+str(it_size)
    #overArea = it_size * it_size

    it_size = kernelSize
    print "Pixel iteration size is " + str(it_size)
    overArea = it_size * it_size

    # CNNs network
    x = base_model.output
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, (1, 1), name='fcn')(x)
    #x = Conv2D(num_classes, (3, 3), padding='same',name='fcn')(x)
    #x = Conv2D(num_classes, (3, 3), strides=(2, 2), name='fcn')(x)
    y = x
    # 2D to point
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    trainModel = Model(inputs=base_model.input, outputs=x, name='Xception_dishes')

    # segmentation
    y = Activation('softmax')(y)
    y = keras.layers.advanced_activations.ThresholdedReLU(theta=judgeThreshold)(y)
    #y = UpSampling2D((fcnUpTimes, fcnUpTimes))(y)
    segModel = Model(inputs=base_model.input, outputs=y, name='Xception_dishes2Fcn')

    if weights_h5 != None and os.path.exists(weights_h5):
        print(weights_h5 +" have loaded!")
        trainModel.load_weights(weights_h5, by_name=True)


    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(trainModel.layers):
    #    print(i, layer.name)
    # exit(0)
    # set the first 15 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for layer in trainModel.layers[0:311]: #block5 top,mix8
    #    layer.trainable = False
    #rgbTo1_layer = trainModel.get_layer(name='rgb2Atomic')
    #rgbTo1_layer.trainable = False

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-5, momentum=0.9, nesterov=True)#-4,-6
    # Let's train the model using opt or sgd
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
    )

    # get train labs info and item name
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

    if evaluate==True:
        print 'evaluate...'
        score = trainModel.evaluate_generator(validation_generator, steps=math.ceil(imgsNumVal / batchSize) * valTimes, workers=4)
        print(score)

    if train==True:
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)
        # Fit the model on the batches generated by datagen.flow().
        trainModel.fit_generator(
            generator=train_generator,
            steps_per_epoch=math.ceil(imgsNumTrain * trainTimes / batchSize),
            epochs=2,
            #callbacks=[early_stopping],
            #validation_data=validation_generator,
            #validation_steps=math.ceil(imgsNumVal*valTimes/batchSize),
            workers=4)

        print 'evaluate...'
        score = trainModel.evaluate_generator(validation_generator, steps=math.ceil(imgsNumVal / batchSize) * valTimes, workers=4)
        print(score)
        #save trained weights
        timeInfo = time.strftime('%m-%d_%H:%M', time.localtime(time.time()))
        trainModel.save_weights(mode+'_'+timeInfo+'_'+str(score[1])+'.h5')

    return trainModel, segModel

#RGB item for Image segmentation

def targetColorInit():

    if num_classes > 0xFFFFFF:
        print ' the categorical num must be smaller than 0xFFFFFF in color segment'
        exit(0)

    targetColor = []
    colorAvg = 0xFFFFFF / (num_classes*4)

    for i in range(0, num_classes):

        color = (i+1) * colorAvg

        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        targetColor.append((r, g, b))
        # print targetColor[i]

    return targetColor

targetColor = targetColorInit()


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
                    if cmp(file, "category.txt") == 0:
                        continue
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
                    if cmp(file, "category.txt") == 0:
                        continue
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

            if imgP[0][i][j][rgbIndex] >= judgeThreshold:
                rgbImg[i, j, :] = targetColor[rgbIndex]#给像素赋值，以示区别
                mylist.append(rgbIndex) #保存class index

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
        h = xr - xl + 1
        w=  yr - yl + 1
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

            if cmp(file, "category.txt") == 0:
                continue

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
                h = xr - xl + 1
                w = yr - yl + 1
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

def compList(a,b):
    if len(a) == len(b):
        c = list(set(a).intersection(set(b))) #交集
        #list(set(a).union(set(b)))       #并集
        #list(set(b).difference(set(a)))  # b中有而a中没有的
        if len(c)==len(a):
            return True
    return False
#a = ['和鱼的', '是的']
#b = ['是的', '和鱼的']
#print compList(a, b)
#exit(0)

def segImgDirforAcc(segPath):
    '''
    images in folder segmentation
    :param segPath: images path
    :return: print and plt show result
    '''
    n=0
    errList = []
    for _, _, files in os.walk(segPath):
        print
        print ("coming " + segPath)
        n = len(files)
        print 'files num is ' +str(n)
        categoryList=[]
        objects = open(segPath+"/category.txt")
        for line in objects:
            line =line[0:-1] #去掉最后的换行符
            categoryList.append(line)
        print 'category is:'
        print categoryList


        for file in files:

            if cmp(file, "category.txt") == 0:
                continue

            print ('predict  ' + file)
            img = loadImage(segPath + '/' + file)
            pred = segModel.predict(img)
            segImg,dishes_info = getRgbImgFromUpsampling(pred)

            # display source img
            objectList=[]
            objectNum = 0
            for dish_k, dish_v in dishes_info.items():  # (dot, name, (xl, yl, xr, yr))
                xl = dish_v[2][0]
                yl = dish_v[2][1]
                xr = dish_v[2][2]
                yr = dish_v[2][3]
                h = xr - xl + 1
                w = yr - yl + 1
                area = h * w
                areaRate = dish_v[0] * 1.0 / overArea
                fullness = dish_v[0] * 1.0 / area
                if areaRate > tageCorrectTh and fullness > degreeCorrectTh:
                    if h > w * 8 or w > h * 8:
                        break
                    objectNum += 1
                    objectList.append(dish_v[1])
                    #print("the %s mostly has found,AreaRate and fullness:(%f,%f)" % (dish_v[1], areaRate, fullness))

            if compList(objectList,categoryList)==False:
                errList.append(file)
                print "seg "+file+' err,NN found:'
                print objectList

    print (segPath + " accury is " + str(n-len(errList)) + '/' + str(n))
    return n, errList

def CalcAccuracySegDir(setsPath,top=3,verbose=1):
    '''
    :param setsPath: images path
    :param top: The maximum probability of top categories
    :param verbose: verbosity mode, 0 or 1.
    :return:recognization accuracy
    '''
    totalNum=0
    totalErrList=[]
    setsPath += '/'
    for _, dirs, _ in os.walk(setsPath):
        for dirName in dirs:
            #for _, _, files in os.walk(setsPath+dirName):
            n, errList = segImgDirforAcc(setsPath+dirName)
            totalNum += n
            totalErrList += errList
            #totalErrList.extend(errList)

    totalAcc = (totalNum-len(totalErrList))*1.0 / totalNum

    print 'seg err files:'
    for errfile in totalErrList:
        print errfile

    print
    print ("total accury is " + str(float('%.4f' % totalAcc )))



def locateImgfile(url):
    '''
    Mark the location of the object with the rectangle of CV2
    :param url: images path
    :return: print and CV2 show image
    '''

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
        h = xr - xl + 1
        w = yr - yl + 1
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