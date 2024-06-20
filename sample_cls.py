# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:55:48 2019

One_Shot_Learning

@author: Don

"""
import itertools

import logging

"""
1.套件区

"""
import gc
import numpy as np
import numpy.random as rng
from PIL import Image
import pickle
import os

import cv2
import time
import joblib
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.layers import Input,GlobalMaxPooling2D,Activation
from keras.layers.core import Lambda, Dense
from keras import backend as K
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator




"""
2.函数区

"""

#定义读取图片与图片前处理
tf.debugging.set_log_device_placement(True)
with tf.device('/CPU:0'):
    def loadimgs(path,n = 0):
        '''
        path => Path of train directory or test directory
        '''
        X=[]
        y = []
        cat_dict = {}
        lang_dict = {}
        curr_y = n
        # we load every alphabet seperately so we can isolate them later
        for alphabet in os.listdir(path):
            print("loading alphabet: " + alphabet) #path/Agaricu...
            lang_dict[alphabet] = [curr_y,None] #lang_dict关键字"alphabet":值"curry_y"
            alphabet_path = os.path.join(path,alphabet) #路径为：xxx/各种字母
            # every letter/category has it's own column in the array, so  load seperately
            for letter in os.listdir(alphabet_path):
                print("loading letter: " + letter)
                cat_dict[curr_y] = (alphabet, letter)
                category_images=[]
                letter_path = os.path.join(alphabet_path, letter) #letter_path是图片目录路径:xxx/各种字母/letter
                # read all the images in the current category
                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path, filename)
                    #image = imread(image_path)
                    try:
                        image = (Image.open(image_path).convert('L')).resize((224,224))
                    except OSError as e:
                        print(e)
                        print("error - category_images:", filename)
                    image = np.asarray(image)
                    image = image.astype('float32')
                    image = image /255
                    category_images.append(image)
                    y.append(curr_y)
                try:
                    X.append(np.stack(category_images))
                # edge case  - last one
                except OSError as e:
                    print(e)
                    print("error - category_images:", filename)
                curr_y += 1
                lang_dict[alphabet][1] = curr_y - 1
        # y = np.vstack(y)
        # X = np.stack(X)
        for item in X:
            print(item.shape)
        print(lang_dict)
        return X,y,lang_dict

    #定义模型
    def get_ResNet50(input_shape):
        """
            Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        """

        # Define the tensors for the two input images
        input = Input(input_shape)

        # Convolutional Neural Network
        model = Sequential()
        model.add(keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,))
        model.add(GlobalMaxPooling2D())
        model.add(Dense(9,activation = 'softmax'))



        # Connect the inputs with the outputs
        ResNet50 = model
        print(ResNet50)

        # return the model
        return ResNet50


    def get_ResNet34(input_shape):
        """
            Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        """

        # Define the tensors for the two input images
        input = Input(input_shape)

        # Convolutional Neural Network
        model = Sequential()
        model.add(keras.applications.ResNet34(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,))
        model.add(GlobalMaxPooling2D())

        model.add(Dense(9, activation='softmax'))
        # Connect the inputs with the outputs
        ResNet34 = model

        # return the model
        return ResNet34

    def get_ResNet101(input_shape):
        """
            Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        """

        # Define the tensors for the two input images
        input = Input(input_shape)

        # Convolutional Neural Network
        model = Sequential()
        model.add(keras.applications.ResNet101(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,))
        model.add(GlobalMaxPooling2D())

        model.add(Dense(9, activation='softmax'))

        # Connect the inputs with the outputs
        ResNet101 = model

        # return the model
        return ResNet101

    #定义 loss function (focal loss)
    def binary_focal_loss(gamma=2, alpha=0.25):
        """
        Binary form of focal loss.

        focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
            where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
         model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)

        def binary_focal_loss_fixed(y_true, y_pred):
            """
            y_true shape need be (None,1)
            y_pred need be compute after sigmoid
            """
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)

            p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
            focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
            return K.mean(focal_loss)
        return binary_focal_loss_fixed

    def dynamic_recall_focal_loss(belta, gamma=2, alpha=0.25):
        """
        dynamic_recall_focal_loss

        DRFL(p_t) = -(1-recall_x)**belta * alpha_t * (1 - p_t)**gamma * log(p_t)
            where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

        Usage:
         model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        alpha = tf.constant(alpha, dtype=tf.float32)
        gamma = tf.constant(gamma, dtype=tf.float32)
        belta = tf.constant(belta, dtype=tf.float32)


        def dynamic_recall_loss_fixed(y_true, y_pred):
            """
            y_true shape need be (None,1)
            y_pred need be compute after sigmoid
            """
            recall_pred = (y_true *y_pred) + K.epsilon() / y_true + K.epsilon()
            y_true = tf.cast(y_true, tf.float32)
            recall_pred = tf.cast(recall_pred, tf.float32)
            alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)

            p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
            class_recall = y_true*recall_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-recall_pred) + K.epsilon()
            dynamic_recall_loss = -K.pow((K.ones_like(y_true)-class_recall),belta) * alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
            return K.mean(dynamic_recall_loss)
        return dynamic_recall_loss_fixed



    #for mushroom classification dataset
    def correct_aug(img):
        # img=np.expand_dims(img, axis=0)
        # k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        # hue_delta = tf.random.uniform([], -0.2, 0.2)
        # tf.image.rot90(img, k=k)
        # tf.image.adjust_hue(img, hue_delta)
        # tf.image.convert_image_dtype(img, dtype=tf.float32)
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        img = datagen.flow(img)



        return img



    #自定义 batch 的产生方式
    def get_batch(batch_size):
        X = Xtrain
        n_classes = len(X) - 1
        # print(n_classes)  #随机选择一个类别
        categories = rng.choice(n_classes,size=(batch_size,),replace=True)
        batch_X = np.zeros((batch_size, 224, 224, 1))
        target = categories
        for i in categories:
            datas = X[i]
            n_samples = datas.shape[0]
            index = rng.randint(0,n_samples)
            temp = datas[index].copy()
            # x = correct_aug(temp)
            batch_X[i, :,:, :] = temp.reshape(224,224,1)
        return batch_X,target



    #自定义 test 方式
    def test_oneshot(model, N, k, verbose = 0): #N=8表验证的图片数，k=80表验证的次数
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        acc_mean = []
        f1_mean = []
        X = Xtrain
        n_classes = len(X) - 1
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
        for i in range(k):
            f1 = []
            acc = []
            inputs, targets = get_batch(N)
            probs = model.predict(inputs)
            for j in range(len(np.unique(targets))):
                predictions = (probs[:, j] > 0.5).astype('int')
                targets = (targets == j).astype('int')
                precision = sum(predictions & targets)/sum(predictions)
                recall = sum(predictions & targets)/sum(targets)
                f1 = 2 * precision +recall/(precision + recall)
                accuary = sum(predictions & targets)/N    #TP+FN
                f1.append(f1)
                acc.append(accuary)
            f1_j = np.mean(f1)
            acc_j = np.mean(acc)
            f1_mean.append(f1_j)
            acc_mean.append(acc_j)
        percent_correct = np.mean(acc_mean)
        percent_f1 = np.mean(f1_mean)
        if verbose:
            logging.info("Got an correct average of {} and f1 score {} of  one-shot learning accuracy \n".format(percent_correct,percent_f1))
        return percent_correct, percent_f1

    #定义训练流程
    def train(n_iter,batch_size,model,evaluate_every,N_way, n_val,model_path):
        train_loss = []
        val_loss_list = []
        T_acc_list = []
        T_f1_list = []
        train_f1_list = []
        train_acc_list = []
        print("Starting training process!")
        print("-------------------------------------")
        t_start = time.time()
        loss = 0
        val_loss = 0
        for i in range(1, n_iter+1):
            (inputs,targets) = get_batch(batch_size)
            loss += model.train_on_batch(inputs, targets)
            del inputs,targets
            gc.collect()
            if i % evaluate_every == 0:
                # print("\n ------------- \n")
                logging.info("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                temp = loss/evaluate_every
                train_loss.append(temp)
                for j in range(evaluate_every):
                    (inputs,targets) = get_batch(batch_size)
                    val_loss += model.test_on_batch(inputs, targets)
                    del inputs,targets
                    gc.collect()
                temp2 = val_loss/evaluate_every
                val_loss_list.append(temp2)
                logging.info("Train Loss: {}, Val Loss: {}".format(temp,temp2))
                loss = 0  #一次验证一次测试？
                val_loss = 0
                T_acc, T_f1 = test_oneshot(model, N_way, n_val,verbose=True)
                T_acc_list.append(T_acc)
                T_f1_list.append((T_f1))
                train_acc,train_f1 = test_oneshot(model, N_way, n_val, verbose = 0)
                train_acc_list.append(train_acc)
                train_f1_list.append(train_f1)
                model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
                logging.info("Got an training set acc average {} accuracy, Val set acc average {} accuracy\n".format(train_acc,T_acc))
                logging.info("Got an training set acc average {} f1, Val set acc average {} f1\n".format(train_f1,T_f1))

        return(train_loss,val_loss_list,T_acc_list,train_acc_list,T_f1_list,train_f1_list)
    """
    3.路径区
    
    """
    #训练资料位置
    val_folder = r"data/test"
    #验证资料位置
    train_folder = r'data/train'
    #训练资料暂存位置(自行设定一个空目录)
    save_path = r'save_data'
    #模型储存路径
    model_path = r'models'

    """
    4.变量区
    
    """
    # Hyper parameters
    evaluate_every = 80
    batch_size = 8
    n_iter = 7000
    N_way = 8 #每次验证时的数量
    n_val = 80 #验证的次数
    learning_rate = 0.8

    """
    5.执行区
    
    """

    #读取训练图片 & 储存到暂存区
    X,y,c=loadimgs(train_folder)
    with open(os.path.join(save_path,"train.pickle"), "wb") as f:
        # pickle.dump((X,c),f,protocol = 4) #memory error
        joblib.dump((X,c),f)
    #读去验证图片 & 储存到暂存区
    Xval,yval,cval=loadimgs(val_folder)
    with open(os.path.join(save_path,"val.pickle"), "wb") as f:
        joblib.dump((Xval,cval),f)
    #从暂存区读取训练资料
    with open(os.path.join(save_path, "train.pickle"), "rb") as f:
        (Xtrain, train_classes) = joblib.load(f)
    #从暂存区读取验证资料
    with open(os.path.join(save_path, "val.pickle"), "rb") as f:
        (Xval, val_classes) = joblib.load(f)
    print("Training alphabets: \n")
    print(list(train_classes.keys()))
    print("Validation alphabets:", end="\n\n")
    print(list(val_classes.keys()))

    #建立模型
    model = get_ResNet50((224, 224, 1))   #get_ResNet34() or get_ResNet101()
    model.summary()
    optimizer = Adam(lr = learning_rate)
    # model.compile(loss=[dynamic_recall_focal_loss(belta=2, alpha=.25, gamma=2)],optimizer=optimizer)
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)],optimizer=optimizer)

    logging.basicConfig(filename='training.log', level=logging.INFO)
    #训练模型
    train_loss,val_loss_list,T_acc_list,T_f1_list,train_acc_list,train_f1_list = train(n_iter,batch_size,model,evaluate_every,
                                                               N_way, n_val,model_path)




