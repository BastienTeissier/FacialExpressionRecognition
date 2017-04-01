from models.vgg import vgg13
from result import emotions
from preprocess import histogramEqualize
from face_capture.process import process_image

import matplotlib.pyplot as plt

import cv2
import numpy as np
import os

model = vgg13()

model.load_weights('model_vgg_13_ck.h5')

def write_label(img, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    cv2.putText(img, label, point, font, font_scale, (255, 0, 0), thickness)

def img_emotions(img, face_cascade=None, eye_cascade=None, rgb=True):
    if rgb:
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        temp = img
    res = []

    if face_cascade == None:
        face_cascade = cv2.CascadeClassifier('face_capture/haarcascade_frontalface_default.xml')
    if eye_cascade == None:
        eye_cascade = cv2.CascadeClassifier('face_capture/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        face = temp[y:y+h, x:x+w]
        face= process_image(face)
        face = histogramEqualize(np.array([[face]]))
        predictions = model.predict(face, batch_size=1)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        write_label(img, (int(x+w)+10,int(y+h)+10), emotions[np.argmax(predictions[0])])
    #plt.imshow(img)
    #plt.show()
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    img = cv2.imread('chirac_mickey.jpg', 1)
    img_emotions(img)
