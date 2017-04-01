from models.vgg import vgg13
from result import emotions
from preprocess import histogramEqualize
from face_capture.process import process_image

from keras.models import Model

import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import math

model = vgg13()

model.load_weights('model_vgg_13_59.h5')

def img_grid(imgs):
    n = int(math.sqrt(len(imgs)))
    a,b = len(imgs[0]), len(imgs[0][0])
    res = np.zeros((a*n,b*n))
    m = 0
    for k in range(len(imgs)):
        x,y = (k%n)*a, (k//n)*b
        print(k)
        print(x,y)
        for i in range(len(imgs[k])):
            for j in range(len(imgs[k][i])):
                res[x+i][y+j] = imgs[k][i][j]
                if imgs[k][i][j]>m:
                    m=imgs[k][i][j]
    return res*255/m

def img_emotions(img, face_cascade=None, eye_cascade=None, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = []

    if face_cascade == None:
        face_cascade = cv2.CascadeClassifier('face_capture/haarcascade_frontalface_default.xml')
    if eye_cascade == None:
        eye_cascade = cv2.CascadeClassifier('face_capture/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    (x,y,w,h) = faces[0]

    face = img[y:y+h, x:x+w]
    face= process_image(face)
    face = histogramEqualize(np.array([[face]]))

    intermediate_layer_model = Model(input=model.input, output=model.layers[7].output)
    output = intermediate_layer_model.predict(face, batch_size=1)
    print(len(output[0]))
    img = img_grid(output[0])
    print(img)
    cv2.imshow('intermediare.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=='__main__':
    img = cv2.imread('chirac.jpg', 1)
    img_emotions(img)
