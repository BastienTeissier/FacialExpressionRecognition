from models import basic, vgg13, vgg16, vgg_face
from preprocess import fer2013, fer2013_light
from result import confusion_matrix, historic

batch_size = 32
nb_classes = 7
nb_epoch = 10
save_weights = True
load_weights = False

# input image dimensions
img_rows, img_cols = 48, 48
# FER2013 is grayscale
img_channels = 1

model = vgg_face('vgg_face.h5')
#(X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()

if load_weights:
    model.load_weights('vgg_face.h5')

history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_validation, Y_validation),
          shuffle=True)

predictions = model.predict(X_test, batch_size=batch_size, verbose=1)

historic(history)

#print(predictions)
confusion_matrix(predictions, Y_test)

if save_weights:
    model.save_weights('vgg_face_keras.h5')
