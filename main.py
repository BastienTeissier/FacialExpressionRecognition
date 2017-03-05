from models import basic
from preprocess import fer2013
from result import confusion_matrix

batch_size = 32
nb_classes = 7
nb_epoch = 1
save_weights = False
load_weights = False

# input image dimensions
img_rows, img_cols = 48, 48
# TFER2013 is grayscale
img_channels = 1

model = basic()
(X_train, Y_train), (X_test, Y_test), (X_validation, Y_validation) = fer2013()

if load_weights:
    model.load_weights('model.h5')

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_validation, Y_validation),
          shuffle=True)

predictions = model.predict(X_test, batch_size=batch_size, verbose=1)

print(predictions)
confusion_matrix(predictions, Y_test)

if save_weights:
    model.save_weights('model.h5')
