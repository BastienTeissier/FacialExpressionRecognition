from models import basic, vgg
from preprocess import fer2013, fer2013_light
from result import confusion_matrix, historic

batch_size = 128
nb_classes = 7
nb_epoch = 100
save_weights = False
load_weights = False

# input image dimensions
img_rows, img_cols = 48, 48
# TFER2013 is grayscale
img_channels = 1

model = vgg()
(X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013_light(2000, 500, 500)

if load_weights:
    model.load_weights('model.h5')

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
    model.save_weights('model.h5')
