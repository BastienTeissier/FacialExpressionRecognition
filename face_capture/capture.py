import cv2
import os

from face_capture.process import process_image

def face_capture(img, face_cascade=None, eye_cascade=None, rgb=False):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = []

    if face_cascade == None:
        face_cascade = cv2.CascadeClassifier('face_capture/haarcascade_frontalface_default.xml')
    if eye_cascade == None:
        eye_cascade = cv2.CascadeClassifier('face_capture/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face)
        if len(eyes)==2:
            res.append(process_image(face))
    return res

if __name__=='__main__':
    imgs = ['faces/'+f for f in os.listdir('faces')]
    faces = []
    for img in imgs:
        img = cv2.imread(img, 0)
        print("Image loaded")
        ret = face_capture(img)
        for r in ret:
            faces.append(r)
    print("Numbers of faces : {}".format(len(faces)))
    k=0
    for face in faces:
        cv2.imwrite('capture/{}.jpg'.format(k), face)
        k+=1
