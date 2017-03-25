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
    #random.shuffle(train)
    X_train = []
    Y_train = []
    for t in train:
        X_train.append(t[0])
        Y_train.append(t[1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('Train set loaded : {} images'.format(len(X_train)))

    validation = np.load('validation_set.npy')
    #random.shuffle(validation)
    X_validation = []
    Y_validation = []
    for v in validation:
        X_validation.append(v[0])
        Y_validation.append(v[1])
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    print('Validation set loaded : {} images'.format(len(X_validation)))

    test = np.load('test_set.npy')
    #random.shuffle(test)
    X_test = []
    Y_test = []
    for t in test:
        X_test.append(t[0])
        Y_test.append(t[1])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print(Y_test)
    print(Y_test[0])

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

    #print(Y_test)
    #print(Y_test[0])

    return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)

def fer2013_light(nb_train, nb_validation, nb_test):
    '''
    Load a part of the fer2013 from .npy files to python and do a little bit of preprocessing
    Return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)
    '''
    train = np.load('train_set.npy')
    #random.shuffle(train)
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

    print(Y_test)
    print(Y_test[0])

    print('Test set loaded : {} images'.format(len(X_test)))

    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, 48, 48)


    X_train = histogramEqualize(X_train)
    X_test = histogramEqualize(X_test)
    X_validation = histogramEqualize(X_validation)

    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    #X_validation = X_validation.astype('float32')

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_validation = np_utils.to_categorical(Y_validation, nb_classes)

    #print(Y_test)
    #print(Y_test[0])

    return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)

def ck():
    '''
    Load the ck dataset from .npy files to python and preprocess
    Return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)
    '''
    train = np.load('train_set_ck_faces.npy')
    #random.shuffle(train)
    X_train = []
    Y_train = []
    for t in train:
        X_train.append(t[0])
        Y_train.append(t[1])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print('Train set loaded : {} images'.format(len(X_train)))

    validation = np.load('val_set_ck_faces.npy')
    #random.shuffle(validation)
    X_validation = []
    Y_validation = []
    for v in validation:
        X_validation.append(v[0])
        Y_validation.append(v[1])
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)

    print('Validation set loaded : {} images'.format(len(X_validation)))

    test = np.load('test_set_ck_faces.npy')
    #random.shuffle(test)
    X_test = []
    Y_test = []
    for t in test:
        X_test.append(t[0])
        Y_test.append(t[1])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print(Y_test)
    print(Y_test[0])

    print('Test set loaded : {} images'.format(len(X_test)))

    print(X_train)
    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48)
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, 48, 48)

    X_train = histogramEqualize(X_train)
    X_test = histogramEqualize(X_test)
    X_validation = histogramEqualize(X_validation)

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_validation = np_utils.to_categorical(Y_validation, nb_classes)

    #print(Y_test)
    #print(Y_test[0])

    return (X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation)

def histogramme_cumule(hist):
    #print(hist)
    hist_c=[hist[0]]
    for i in range(1,len(hist)):
        hist_c.append(float(hist_c[i-1]+hist[i]))
    hist_c = np.array(hist_c)
    #print("Histogramme cumulé")
    #print(len(hist))
    #print(len(hist_c))
    return hist_c/hist_c[len(hist_c)-1]*255

def equalize(img):
    hist = np.histogram(img, bins=np.arange(257))
    #print(hist[0])
    hist_cum = histogramme_cumule(hist[0])
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[0][i][j] = hist_cum[int(img[0][i][j])]
    return img

def histogramEqualize(X):
    return np.array([equalize(img) for img in X])
