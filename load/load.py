import pandas as pd
import numpy as np
import cv2

import os
import random

from face_capture.process import process_image

def load_fer2013(path):
    '''
    Load the fer2013 datasets from the csv file with the form (emotion, pixels, Usage)
    Datas are save in 3 separate files according to their use :
    - train_set.npy
    - validation_set.npy
    - test_set.npy
    Each file containing a list of tuples (data, label)
    '''
    df = pd.read_csv(path)
    train_set = []
    test_set = []
    validation_set = []
    for k in range(len(df)):
        # CSVs contain only string, we split it in an array
        if k%1000 == 0:
            print(k)
        pixels = df.loc[k]['pixels'].split(' ')

        img = np.zeros((48,48))
        # Reconstruct the image/matrix from the pixels array
        for i in range(len(pixels)):
            img[i//48][i%48] = int(pixels[i])
        if df.loc[k]['Usage'] == 'PrivateTest':
            test_set.append((np.array(img), df.loc[k]['emotion']))
        elif df.loc[k]['Usage'] == 'PublicTest':
            validation_set.append((np.array(img), df.loc[k]['emotion']))
        else:
            train_set.append((np.array(img), df.loc[k]['emotion']))
    np.save('train_set.npy', train_set)
    np.save('test_set.npy', test_set)
    np.save('validation_set.npy', validation_set)
'''
#
# Does not work because of missing image in FERPlus
#
def load_plus():
    df = pd.read_csv('fer2013.csv')
    df_train = pd.read_csv('train.csv')
    df_val = pd.read_csv('val.csv')
    df_test = pd.read_csv('test.csv')

    labels = {}

    train_labels = {}
    print("Loading training labels")
    for k in range(len(df_train)):
        if k%1000 == 0:
            print(k)
        temp = []
        for i in range(2,len(df_train.loc[k])):
            temp.append(df_train.loc[k][i])
        train_labels[int(df_train.loc[k][0].replace('.png','').replace('fer',''))] = np.array(temp)
        labels[int(df_train.loc[k][0].replace('.png','').replace('fer',''))] = np.array(temp)

    val_labels = {}
    print("Loading validation labels")
    for k in range(len(df_val)):
        if k%1000 == 0:
            print(k)
        temp = []
        for i in range(2,len(df_val.loc[k])):
            temp.append(df_val.loc[k][i])
        val_labels[int(df_val.loc[k][0].replace('.png','').replace('fer',''))] = np.array(temp)
        labels[int(df_val.loc[k][0].replace('.png','').replace('fer',''))] = np.array(temp)

    test_labels = {}
    print("Loading test labels")
    for k in range(len(df_test)):
        if k%1000 == 0:
            print(k)
        temp = []
        for i in range(2,len(df_test.loc[k])):
            temp.append(df_test.loc[k][i])
        train_labels[int(df_test.loc[k][0].replace('.png','').replace('fer',''))] = np.array(temp)
        labels[int(df_test.loc[k][0].replace('.png','').replace('fer',''))] = np.array(temp)

    train_set = []
    test_set = []
    validation_set = []
    print("Loading data")
    for k in range(len(df)):
        # CSVs contain only string, we split it in an array
        if k%1000 == 0:
            print(k)
        print(k)
        pixels = df.loc[k]['pixels'].split(' ')

        img = np.zeros((48,48))
        # Reconstruct the image/matrix from the pixels array
        for i in range(len(pixels)):
            img[i//48][i%48] = int(pixels[i])
        if df.loc[k]['Usage'] == 'PrivateTest':
            test_set.append((np.array(img), labels[k+1]))
        elif df.loc[k]['Usage'] == 'PublicTest':
            validation_set.append((np.array(img), labels[k+1]))
        else:
            train_set.append((np.array(img), labels[k+1]))
    print("Saving...")
    np.save('train_set_plus.npy', train_set)
    np.save('test_set_plus.npy', test_set)
    np.save('validation_set_plus.npy', validation_set)
    print("Done")
'''

def load_ck(src):
    emotions = {"angry":0, "disgust":1, "fear":2, "happy":3, "sad":4, "surprise":5, "neutral":6}
    X = [[] for i in range(7)]
    for folder in os.listdir(src):
        e = folder.split('_')[0]
        if e!= 'contempt':
            emotion = emotions[e]
            print(folder)
            for f in os.listdir(src+'/'+folder):
                try:
                    img = cv2.imread(src+'/'+folder+'/'+f,0)
                    img = process_image(img)
                    X[emotion].append(img)
                except:
                    print("Error with the image")
    train, val, test = [],[],[]
    for i in range(len(X)):
        random.shuffle(X[i])
        for e in range(int(8*len(X[i])/10)):
            train.append((np.array(X[i][e]),i))
        for e in range(int(8*len(X[i])/10), int(9*len(X[i])/10)):
            val.append((np.array(X[i][e]),i))
        for e in range(int(9*len(X[i])/10), len(X[i])):
            test.append((np.array(X[i][e]),i))
    np.save('train_set_ck.npy', train)
    np.save('val_set_ck.npy', val)
    np.save('test_set_ck.npy', test)


if __name__ == '__main__':
    load_ck('result')
