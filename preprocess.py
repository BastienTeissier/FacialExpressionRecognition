# -*- coding: utf-8 -*-

import random
from keras.utils import np_utils
import numpy as np
import cv2

# This issue again may be a constant.py file
nb_classes = 7

def fer2013():
    '''
    Load the fer2013 from .npy files to python and do a little bit of preprocessing
    Return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)
    '''
    train = np.load('train_set.npy')
    X_train = []
    Y_train = []
    for t in train:
        X_train.append(t[0])
        Y_train.append(t[1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('Train set loaded : {} images'.format(len(X_train)))

    validation = np.load('validation_set.npy')
    X_validation = []
    Y_validation = []
    for v in validation:
        X_validation.append(v[0])
        Y_validation.append(v[1])
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    print('Validation set loaded : {} images'.format(len(X_validation)))

    test = np.load('test_set.npy')
    X_test = []
    Y_test = []
    for t in test:
        X_test.append(t[0])
        Y_test.append(t[1])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print('Test set loaded : {} images'.format(len(X_test)))

    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, 48, 48)


    X_train = histogramEqualize(X_train)
    X_test = histogramEqualize(X_test)
    X_validation = histogramEqualize(X_validation)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_validation = np_utils.to_categorical(Y_validation, nb_classes)

    return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)

def fer2013_light(nb_train, nb_validation, nb_test):
    '''
    Load a part of the fer2013 from .npy files to python and do a little bit of preprocessing
    Return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)
    '''
    train = np.load('train_set.npy')
    X_train = []
    Y_train = []
    for t in train[:nb_train]:
        X_train.append(t[0])
        Y_train.append(t[1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('Train set loaded : {} images'.format(len(X_train)))

    validation = np.load('validation_set.npy')
    random.shuffle(validation)
    X_validation = []
    Y_validation = []
    for v in validation[:nb_validation]:
        X_validation.append(v[0])
        Y_validation.append(v[1])
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    print('Validation set loaded : {} images'.format(len(X_validation)))

    test = np.load('test_set.npy')
    random.shuffle(test)
    X_test = []
    Y_test = []
    for t in test[:nb_test]:
        X_test.append(t[0])
        Y_test.append(t[1])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print('Test set loaded : {} images'.format(len(X_test)))

    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, 48, 48)

    X_train = histogramEqualize(X_train)
    X_test = histogramEqualize(X_test)
    X_validation = histogramEqualize(X_validation)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_validation = np_utils.to_categorical(Y_validation, nb_classes)

    return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)

def ck():
    '''
    Load the ck dataset from .npy files to python and preprocess
    Return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)
    '''
    train = np.load('train_set_ck_faces.npy')
    X_train = []
    Y_train = []
    for t in train:
        X_train.append(t[0])
        Y_train.append(t[1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('Train set loaded : {} images'.format(len(X_train)))

    validation = np.load('val_set_ck_faces.npy')
    X_validation = []
    Y_validation = []
    for v in validation:
        X_validation.append(v[0])
        Y_validation.append(v[1])
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    print('Validation set loaded : {} images'.format(len(X_validation)))

    test = np.load('test_set_ck_faces.npy')
    X_test = []
    Y_test = []
    for t in test:
        X_test.append(t[0])
        Y_test.append(t[1])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print('Test set loaded : {} images'.format(len(X_test)))

    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, 48, 48)

    X_train = histogramEqualize(X_train)
    X_test = histogramEqualize(X_test)
    X_validation = histogramEqualize(X_validation)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_validation = np_utils.to_categorical(Y_validation, nb_classes)

    return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)

def histogramme_cumule(hist):
    hist_c=[hist[0]]
    for i in range(1,len(hist)):
        hist_c.append(float(hist_c[i-1]+hist[i]))
    hist_c = np.array(hist_c)
    return hist_c/hist_c[len(hist_c)-1]*255

def equalize(img):
    hist = np.histogram(img, bins=np.arange(257))
    hist_cum = histogramme_cumule(hist[0])
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[0][i][j] = hist_cum[int(img[0][i][j])]
    return img

def histogramEqualize(X):
    return np.array([equalize(img) for img in X])

def gaussianFilter(X):
    return np.array([cv2.GaussianBlur(img,(3,3),0) for img in X])

def median_substraction(X):
    median_substraction = [[0 for i in range(len(X[0]))] for i in range (len(X))]
    #Top left corner
    neighbourhood = [X[0][0], X[0][1], X[1][0], \
                    X[1][1]]
    neighbourhood.sort()
    median_substraction[0][0] += X[0][0] - (neighbourhood[1]+neighbourhood[2])/2
    print(median_substraction[0][0])
    #Top right corner
    neighbourhood = [X[0][len(X[0])-1], X[0][len(X[0])-2], X[1][len(X[0])-1], \
                    X[1][len(X[0])-2]]
    neighbourhood.sort()
    median_substraction[0][len(X[0])-1] += X[0][len(X[0])-1] - (neighbourhood[1]+neighbourhood[2])/2
    print(median_substraction[0][len(X[0])-1])
    #Bottom left corner
    neighbourhood = [X[len(X)-1][0], X[len(X)-1][1], X[len(X)-2][0], \
                    X[len(X)-2][1]]
    neighbourhood.sort()
    median_substraction[len(X)-1][0] += X[len(X)-1][0] - (neighbourhood[1]+neighbourhood[2])/2
    print(median_substraction[len(X)-1][0])
    #Bottom right corner
    neighbourhood = [X[len(X)-1][len(X[0])-1], X[len(X)-1][len(X[0])-2], \
                    X[len(X)-2][len(X[0])-1], X[len(X)-2][len(X[0])-2]]
    neighbourhood.sort()
    median_substraction[len(X)-1][len(X[0])-1] += X[len(X)-1][len(X[0])-1] - (neighbourhood[1]+neighbourhood[2])/2
    print(median_substraction[len(X)-1][len(X[0])-1])
    #Horizontal borders
    for i in range(1,len(X)-1):
        neighbourhood = [X[i-1][0], X[i-1][1], X[i][0], X[i][1], X[i+1][0], \
                        X[i+1][1]]
        neighbourhood.sort()
        median_substraction[i][0] += X[i][0] - (neighbourhood[2]+neighbourhood[3])/2
        neighbourhood = [X[i-1][len(X[0])-1], X[i-1][len(X[0])-1], X[i][len(X[0])-1], \
                        X[i][len(X[0])-1], X[i+1][len(X[0])-1],X[i+1][len(X[0])-1]]
        neighbourhood.sort()
        median_substraction[i][len(X[0])-1] += X[i][len(X[0])-1] - (neighbourhood[2]+neighbourhood[3])/2
    #Vertical borders
    for k in range(1, len(X[0])-1):
        neighbourhood = [X[0][k-1], X[0][k-1], X[0][k-1], X[1][k-1], X[1][k], \
                        X[1][k+1]]
        neighbourhood.sort()
        median_substraction[0][k] += X[0][k] - (neighbourhood[2]+neighbourhood[3])/2
        neighbourhood = [X[len(X)-1][k-1], X[len(X)-1][k-1], X[len(X)-1][k-1], \
                        X[len(X)-2][k-1], X[len(X)-2][k], X[len(X)-2][k+1]]
        neighbourhood.sort()
        median_substraction[len(X)-2][k] += X[len(X)-2][k] - (neighbourhood[2]+neighbourhood[3])/2
    print(median_substraction[0])
    print(median_substraction[len(X)-2])
    #Center
    for i in range(1,len(X)-1):
        for k in range(1, len(X[0])-1):
            neighbourhood = [X[i-1][k-1], X[i-1][k], X[i-1][k+1], X[i][k-1],\
                            X[i][k], X[i][k+1], X[i+1][k-1], X[i+1][k],\
                            X[i+1][k+1]]
            neighbourhood.sort()
            median_substraction[i][k] += X[i][k] - neighbourhood[4]
    print(median_substraction)
    return np.array(median_substraction)



if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()
    gaussian = gaussianFilter(X_train)
    median = median_substraction(gaussian[0][0])
    cv2.imwrite('img.jpg', gaussian[0][0])
    cv2.imwrite('img1.jpg', X_train[0][0])
    cv2.imwrite('img2.jpg', median)
