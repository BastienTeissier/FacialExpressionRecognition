from models import basic, vgg13, vgg16
from preprocess import fer2013, fer2013_light
from result import emotions

from face_capture.capture import face_capture

import cv2
import numpy as np
import os

batch_size = 128
nb_classes = 7
nb_epoch = 10
save_weights = False
load_weights = True

# input image dimensions
img_rows, img_cols = 48, 48
# FER2013 is grayscale
img_channels = 1

imgs = ['face_capture/faces/'+f for f in os.listdir('face_capture/faces')]

faces = []

for img in imgs:
    img = cv2.imread(img, 0)
    print("Image loaded")
    ret = face_capture(img)
    for r in ret:
        faces.append(ret)

X_test = faces
'''
for face in faces:
    print(face)
    for i in range(len(face)):
        temp_rows = []
        for j in range(len(face[i])):
            temp_rows.append(int(face[i][j]))
        temp_list.append(list(temp_rows))
    X_test.append(np.array(temp_list))
'''

#X_test = np.array([np.array(X_test)])
#X_test/=255
print(X_test)

model = vgg13()

model.load_weights('model_vgg_13_59.h5')

predictions = model.predict(X_test, batch_size=batch_size, verbose=1)

for i in range(len(predictions)):
    print(emotions[np.argmax(predictions[i])])
    #cv2.imshow(X_test[0][0][i])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
